# "Validate" xvectors by training a simple siamese network
# to say 1 if both samples are from same speaker, 0 otherwise
import argparse
import os
import pickle
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from torch_utils.models import SiameseWithDiscriminator
from utils.plot_utils import plot_score_distribution, plot_det, compute_eer
from utils.data_loading import load_asv_per_speaker, load_enroll_data, load_trial_data, Trial

parser = argparse.ArgumentParser("Test-train simple Siamese ASV system via mutual information training")
parser.add_argument("filelist", help="Location of filelist used for training (or enrollment)")
parser.add_argument("feature_directory", help="Location where xvectors are stored")
parser.add_argument("model", help="Path to the model")
parser.add_argument("operation", choices=["train", "eval", "bulk-eval"])
parser.add_argument("--trial-list", help="Trial list used for ASV evaluation.")
parser.add_argument("--output", type=str, help="While where encodings or scores will be written.")
parser.add_argument("--save-every-epochs", type=int, default=None, help="How often to save snapshots of the model.")
parser.add_argument("--models", type=str, nargs="*", help="Set of models used for bulk evaluation")
parser.add_argument("--load-model", type=str, help="Location of model to load before training")

# Training setup
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Yer good olde learning rate")
parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam"],
                    help="Optimizer to use")
parser.add_argument("--preprocessing", type=str, choices=["center", "lda"], nargs="*",
                    help="Preprocessing steps to run on data.")
parser.add_argument("--lda-dim", type=int, default=150, help="Target dimensionality for LDA. If None, do not use.")
parser.add_argument("--dropout", type=float, default=0.0, help="Amount of dropout per discriminator layer.")
parser.add_argument("--l2-weight", type=float, default=0.0, help="L2 regularization weight used (Torch calls it weight-decay)")

XVECTOR_DIM = 512
BATCH_SIZE = 64


def create_network(input_dim, args):
    feature_network = nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
    ).cuda()
    model = SiameseWithDiscriminator(
        feature_size=256,
        output_size=1,
        feature_network=feature_network,
        num_discriminator_layers=2,
        num_discriminator_units=512,
        discriminator_dropout=args.dropout,
    ).cuda()

    return model, feature_network


def create_sample_batch(speaker_data, batch_size, vector_dim):
    """
    Return torch tensors ((input1, input2), target) for siamese
    network training (for mutual information).

    Constructs each batch to have roughly equal amount of target
    and non-target samples to keep training balanced
    """
    input1_tensor = torch.zeros(batch_size, vector_dim).float()
    input2_tensor = torch.zeros(batch_size, vector_dim).float()
    target_tensor = torch.zeros(batch_size, 1).float()

    speaker_ids = list(speaker_data.keys())
    for i in range(batch_size):
        # Throw a coin if we add target or non-target sample
        input1 = None
        input2 = None
        
        target = None
        if random.random() < 0.5:
            # Target sample:
            #   Pick speaker, and then pick two separate vectors
            #   from that speaker
            random_speaker_id = random.choice(speaker_ids)
            speaker_vectors = speaker_data[random_speaker_id]
            # random.sample takes two unique vectors 
            input1, input2 = random.sample(speaker_vectors, 2)
            target = 1
        else:
            # Non-target sample
            #  Pick two different speakers and one vector from
            #  each
            speaker1, speaker2 = random.sample(speaker_ids, 2)
            input1 = random.choice(speaker_data[speaker1])
            input2 = random.choice(speaker_data[speaker2])
            target = 0

        # Put sampled vectors to batch
        input1_tensor[i] = torch.from_numpy(input1).float()
        input2_tensor[i] = torch.from_numpy(input2).float()
        target_tensor[i, 0] = target

    input1_tensor = input1_tensor.cuda()
    input2_tensor = input2_tensor.cuda()
    target_tensor = target_tensor.cuda()

    return ((input1_tensor, input2_tensor), target_tensor)


def data_preprocessing(vectors_pooled, args, labels_pooled=None, preprocessing_parameters=None):
    """
    Do preprocessing on data:
        - Centering (minus mean, as in x-vector paper)
        - LDA (if enabled)

    Parameters:
        vectors_pooled (ndarray): Matrix of NxD of all available speaker
                                  vectors
        args: Arguments from ArgumentParser
        labels_pooled (ndarray, optional): Vector of N, labeling each vector in
                                           vectors_pooled to one class (speaker).
                                           Not needed for evaluating.
        preprocessing_parameters (dict, optional): If provided, use these
                                             parameters to do the
                                             preprocessing rather than
                                             learning new parameters
    Returns:
        vectors_pooled (ndarray): Transformed/preprocessed vectors_pooled
        preprocessing_parameters (dict): Dictionary mapping processing names
                                   to objects that contain parameters
                                   to do that preprocessing (e.g.
                                   means for centering).
    """

    # This will be filled with parameters to be learned
    # if preprocessing_parameters is not given
    return_preprocessing_parameters = None if preprocessing_parameters is not None else {}

    # If no preprocessing steps are set, just return
    if args.preprocessing is None:
        return vectors_pooled, return_preprocessing_parameters

    # Centering
    if "centering" in args.preprocessing:
        print("Centering vectors...")
        mean_vector = None

        if preprocessing_parameters is not None:
            mean_vector = preprocessing_parameters["centering_mean"]
        else:
            mean_vector = np.mean(vectors_pooled, axis=0)
        vectors_pooled -= mean_vector[None, :]

        # Save learned mean vector if we were not provided with one
        if preprocessing_parameters is None:
            return_preprocessing_parameters["centering_mean"] = mean_vector

    # LDA dimensionality reducation
    if "lda" in args.preprocessing:
        print("Dimensionality reduction with LDA...")
        # Check if we have been provided with learned parameters
        if preprocessing_parameters is not None:
            lda = preprocessing_parameters["lda"]
            vectors_pooled = lda.transform(vectors_pooled)
        else:
            lda = LinearDiscriminantAnalysis(n_components=args.lda_dim)
            vectors_pooled = lda.fit_transform(vectors_pooled, labels_pooled)
            return_preprocessing_parameters["lda"] = lda

    return vectors_pooled, return_preprocessing_parameters


def _train_data_preprocessing(speaker_data, args, **kwargs):
    """
    data_preprocessing but for main_train function where
    data is stored in a dictionary per speaker.
    Does necessary data transformations to call
    data_preprocessing
    """
    # Turn dictionary into one big array for data_preprocessing
    unique_speaker_ids = list(speaker_data.keys())
    labels_pooled = []
    vectors_pooled = []

    for i, speaker_id in enumerate(unique_speaker_ids):
        speaker_vectors = np.array(speaker_data[speaker_id])
        vectors_pooled.append(speaker_vectors)
        labels_pooled.append(np.zeros((len(speaker_vectors),)) + i)

    labels_pooled = np.concatenate(labels_pooled, axis=0)
    vectors_pooled = np.concatenate(vectors_pooled, axis=0)

    # Call preprocessing
    vectors_pooled, return_preprocessing_parameters = data_preprocessing(
        vectors_pooled, args, labels_pooled=labels_pooled, **kwargs
    )

    # Turn data back to dictionary, similar to what we got in
    new_speaker_data = {}

    for i, speaker_id in enumerate(unique_speaker_ids):
        speaker_vectors = vectors_pooled[labels_pooled == i]
        # Turn matrix into list of vectors
        speaker_vectors = [speaker_vectors[j,:] for j in range(len(speaker_vectors))]

        new_speaker_data[speaker_id] = speaker_vectors

    return new_speaker_data, return_preprocessing_parameters


def _eval_data_preprocessing(trial_list, args, **kwargs):
    """
    data_preprocessing but for main_eval function where
    data is stored in trials.
    Does necessary data transformations to call
    data_processing.
    """
    # First process trial list
    # Turn trial list into big arrays
    trial_trial_features = []
    trial_test_features = []

    for trial in trial_list:
        trial_trial_features.append(trial.trial_features)
        trial_test_features.append(trial.test_features)

    # Sanity check:
    # Either we have test_features for all trials or
    # all of them are Nones
    test_features_is_none = list(map(lambda x: x is None, trial_test_features))
    all_test_features_are_none = all(test_features_is_none)
    if any(test_features_is_none) and not all_test_features_are_none:
        raise RuntimeError("Only some of the test_features were None, but not all")

    trial_trial_features = np.array(trial_trial_features)
    if not all_test_features_are_none:
        trial_test_features = np.array(trial_test_features)

    # Preprocessing
    trial_trial_features, _ = data_preprocessing(trial_trial_features, args, **kwargs)
    if not all_test_features_are_none:
        trial_test_features, _ = data_preprocessing(trial_test_features, args, **kwargs)

    # Turn back into Trials
    new_trial_list = []

    for i in range(len(trial_trial_features)):
        original_trial = trial_list[i]
        new_trial_features = trial_trial_features[i]
        # test_features are here either Nones or vectors,
        # depending on if they were originally provided
        new_test_features = trial_test_features[i]

        new_trial_list.append(Trial(
            trial_features=new_trial_features,  
            claimed_identity=original_trial.claimed_identity,
            test_features=new_test_features,
            is_target=original_trial.is_target,
            origin=original_trial.origin
        ))

    return new_trial_list, None


def main_train(args):
    if args.load_model is not None and args.preprocessing is not None:
        raise NotImplementedError("Support for loading model and preprocessing not done yet")

    speaker_data = load_asv_per_speaker(args.filelist, args.feature_directory)

    num_samples = sum([len(speaker_features) for speaker_features in speaker_data.values()])
    iterations = int(args.epochs * (num_samples // BATCH_SIZE))
    save_every_iterations = None
    if args.save_every_epochs:
        save_every_iterations = (args.save_every_epochs * (num_samples // BATCH_SIZE))

    # Preprocess data
    speaker_data, preprocessing_parameters = _train_data_preprocessing(speaker_data, args)

    # Check the size of the vectors now
    vector_dim = speaker_data[list(speaker_data.keys())[0]][0].shape[0]

    model, feature_network = create_network(vector_dim, args)

    # Load model if so desired
    if args.load_model is not None:
        state_dict = torch.load(args.load_model)
        preprocessing_parameters_saved = {
            "centering_mean": state_dict.pop("centering_mean", None),
            "lda": state_dict.pop("lda", None)
        }
        model.load_state_dict(state_dict)

    optimizer_class = None
    if args.optimizer == "adam":
        optimizer_class = torch.optim.Adam
    elif args.optimizer == "sgd":
        optimizer_class = torch.optim.SGD

    optimizer = optimizer_class(model.parameters(), weight_decay=args.l2_weight, lr=args.lr)

    progress_bar = tqdm(total=iterations)
    loss_deque = deque(maxlen=100)
    for update_i in range(iterations):
        # Inputs is a tuple of two matrices of inputs
        # to the siamese network.
        # Targets is {0,1} labels
        inputs, targets = create_sample_batch(
            speaker_data,
            BATCH_SIZE,
            vector_dim
        )

        predictions = model(*inputs)
        # Sigmoid prediction
        predictions = torch.sigmoid(predictions)

        # Cross-entropy loss
        loss = F.binary_cross_entropy(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_deque.append(loss.item())
        average_loss = np.mean(loss_deque)
        progress_bar.update(1)
        progress_bar.set_description("Loss: {:2.5f}".format(average_loss))
        progress_bar.refresh()

        # Check if we should save a snapshot of the agent
        if save_every_iterations is not None and ((update_i + 1) % save_every_iterations) == 0:
            # Compute number of epochs passed
            epochs_passed = round((update_i / iterations) * args.epochs)
            # Save model and required pretraining
            state_dict = model.state_dict()
            state_dict.update(preprocessing_parameters)
            torch.save(state_dict, args.model + "_epochs_{}".format(epochs_passed))

    # Save model and required pretraining
    state_dict = model.state_dict()
    # Add pretraining parameters to the state_dict:
    # This way the eval code will throw errors if not all
    # pretraining parameters are handled (removed)
    state_dict.update(preprocessing_parameters)

    torch.save(state_dict, args.model)


def score_trials(trial_list, model, enroll_data=None):
    """
    Score trials with model.

    Returns three lists: target_scores, nontarget_scores and scores,
    where scores is a list of scores in same order as trials
    """
    target_scores = []
    nontarget_scores = []
    scores = []

    # Batch up several trials before
    # moving to torch and running predictions
    batch_input1 = []
    batch_input2 = []
    batch_trials = []

    for i in tqdm(range(len(trial_list)), desc="score", leave=False):
        trial = trial_list[i]
        batch_input1.append(trial.trial_features)

        trial_features = torch.from_numpy(trial.trial_features).float().cuda()

        enroll_features = None
        if enroll_data is not None:
            # Get enrolled features based on speaker-id
            enroll_features = enroll_data[trial.claimed_identity]
        else:
            # Use features loaded in the Trial
            enroll_features = trial.test_features
        batch_input2.append(enroll_features)
        batch_trials.append(trial)

        # If enough samples in the batch (or final iteration),
        # do predictions
        if len(batch_trials) >= 64 or i == (len(trial_list) - 1):
            batch_input1_torch = torch.from_numpy(np.stack(batch_input1)).float().cuda()
            batch_input2_torch = torch.from_numpy(np.stack(batch_input2)).float().cuda()
            prediction_scores = model(batch_input1_torch, batch_input2_torch).cpu().detach().numpy().ravel()

            for score_trial, score in zip(batch_trials, prediction_scores):
                if score_trial.is_target:
                    target_scores.append(score)
                else:
                    nontarget_scores.append(score)
                scores.append(score)

            batch_input1.clear()
            batch_input2.clear()
            batch_trials.clear()

    return target_scores, nontarget_scores, scores


def main_eval(args):
    """
    Get different classification errors for the ASV task.
    """

    # Load model, which also contains the preprocessing steps
    state_dict = torch.load(args.model)
    # Get the preprocessing parameters
    preprocessing_parameters = {
        "centering_mean": state_dict.pop("centering_mean", None),
        "lda": state_dict.pop("lda", None)
    }

    # If we are scoring (gathering outputs), we possibly want
    # to score all files, including spoof ones
    skip_spoof = True if args.output is None else False
    trial_list = load_trial_data(args.trial_list, args.feature_directory, skip_spoof=skip_spoof)

    # Preprocess trial data
    trial_list, _ = _eval_data_preprocessing(
        trial_list, args, preprocessing_parameters=preprocessing_parameters
    )

    # Not all datasets use enrolling speakers
    # (they test one sample against another).
    # Check if Trials have "test_features". If they do
    # they already contain the test features we need
    # and no enrollment is required
    needs_enroll_data = trial_list[0].test_features is None

    enroll_data = None
    if needs_enroll_data:
        enroll_data = load_enroll_data(args.filelist, args.feature_directory)

        # Data is in same format as with training data, so we can use that
        # function for preprocessing
        enroll_data, _ = _train_data_preprocessing(
            enroll_data, args, preprocessing_parameters=preprocessing_parameters
        )

        # Enroll speakers by averaging x-vectors
        averaged_enroll_data = {}
        for speaker_id, features in enroll_data.items():
            average_x_vector = np.mean(np.stack(features), axis=0)
            averaged_enroll_data[speaker_id] = average_x_vector

        enroll_data = averaged_enroll_data

    # Check size (dimensionality) of the input vectors
    vector_dim = trial_list[0].trial_features.shape[0]

    # Create the actual model
    model, feature_network = create_network(vector_dim, args)

    model.load_state_dict(state_dict)
    model.eval()

    target_scores, nontarget_scores, scores = score_trials(trial_list, model, enroll_data)

    # Save individual trial scores if we so desire
    if args.output is not None:
        output_lines = []
        for trial, score in zip(trial_list, scores):
            is_target = trial.is_target
            system = trial.origin
            output_lines.append("{} {} {}".format(is_target, score, system))

        with open(args.output, "w") as f:
            f.write("\n".join(output_lines))

    target_scores = np.array(target_scores)
    nontarget_scores = np.array(nontarget_scores)

    print("Total target scores:    {}".format(len(target_scores)))
    print("Total nontarget scores: {}".format(len(nontarget_scores)))
    if skip_spoof:
        eer = compute_eer(target_scores, nontarget_scores)
        print("EER:                    {}".format(eer))
    else:
        print("EER disabled (spoof samples included in ASV trials)")


def main_bulk_eval(args):
    """
    Evaluate bunch of models on the same dataset
    """

    if args.preprocessing is not None:
        raise NotImplementedError("Preprocessing not implemented yet")

    skip_spoof = False
    trial_list = load_trial_data(args.trial_list, args.feature_directory, skip_spoof=skip_spoof)

    # Not all datasets use enrolling speakers
    # (they test one sample against another).
    # Check if Trials have "test_features". If they do
    # they already contain the test features we need
    # and no enrollment is required
    needs_enroll_data = trial_list[0].test_features is None

    enroll_data = None
    if needs_enroll_data:
        enroll_data = load_enroll_data(args.filelist, args.feature_directory)

        # Enroll speakers by averaging x-vectors
        averaged_enroll_data = {}
        for speaker_id, features in enroll_data.items():
            average_x_vector = np.mean(np.stack(features), axis=0)
            averaged_enroll_data[speaker_id] = average_x_vector

        enroll_data = averaged_enroll_data

    # Create the actual model
    model, feature_network = create_network(XVECTOR_DIM, args)

    # Check how long filenames we have so we can do prettier printing
    longest_filename = max(map(len, args.models))
    print_template = "{:<%d}  {:>6}" % (longest_filename)
    # Print header
    print(print_template.format("", "EER"))

    # Loop over different models, load their parameters
    # and use them to score files

    for model_path in args.models:
        state_dict = torch.load(model_path)
        # Remove preprocessing elements
        preprocessing_parameters = {
            "centering_mean": state_dict.pop("centering_mean", None),
            "lda": state_dict.pop("lda", None)
        }
        model.load_state_dict(state_dict)
        model.eval()

        target_scores, nontarget_scores, scores = score_trials(trial_list, model, enroll_data)

        # If skip_spoof is False, do not print out
        # EER as it does not make sense to compute
        # ASV EER with spoof samples
        if skip_spoof:
            target_scores = np.array(target_scores)
            nontarget_scores = np.array(nontarget_scores)
            eer = compute_eer(target_scores, nontarget_scores)
            print(print_template.format(model_path, round(eer, 4)))
        else:
            print(print_template.format(model_path, "Spoof"))

        # Save individual trial scores if we so desire
        if args.output is not None:
            output_lines = []
            for trial, score in zip(trial_list, scores):
                is_target = trial.is_target
                system = trial.origin
                output_lines.append("{} {} {}".format(is_target, score, system))

            filename = os.path.join(args.output, os.path.basename(model_path) + ".txt.asv_scores")
            with open(filename, "w") as f:
                f.write("\n".join(output_lines))

if __name__ == "__main__":
    args = parser.parse_args()
    if args.operation == "train":
        main_train(args)
    elif args.operation == "eval":
        main_eval(args)
    else:
        main_bulk_eval(args)
