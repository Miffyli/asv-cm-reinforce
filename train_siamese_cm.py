# Train a siamese network structure to distinguish
# between spoof and bonafide samples with the 
# maximization of mutual information:
#  [1] https://arxiv.org/abs/1812.00271

# The idea here is to use average
# features of bona-fide and spoof samples
# as the comparison point.

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

from torch_utils.models import XVectorLikeNetwork, SiameseWithDiscriminator, ConvolutionFeatureProcessor
from utils.plot_utils import plot_score_distribution, plot_det, compute_eer
from utils.data_loading import load_cm_train_data, load_cm_trial_data

parser = argparse.ArgumentParser("Test-train simple siamese CM system via mutual information training")
parser.add_argument("filelist", help="Location of ASVSpoof2019 filelist used for training or for eval")
parser.add_argument("feature_directory", help="Location where features are stored")
parser.add_argument("model", help="Path to the model")
parser.add_argument("operation", choices=["train", "eval", "encode", "bulk-eval"])
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--output", type=str, help="Where to store encodings or scores.")
parser.add_argument("--save-every-epochs", type=int, default=None, help="How often to save snapshots of the model.")
parser.add_argument("--models", type=str, nargs="*", help="Set of models used for bulk evaluation")
parser.add_argument("--feature-parameters", type=str, help="Load parameters for the feature network")
parser.add_argument("--freeze-feature", action="store_true", help="Freeze feature network (siamese) parameters")

# Training configs
parser.add_argument("--stats", type=str, choices=["mean", "std", "max"], default=["mean"], nargs="+",
                    help="Statistics to include pooling over features.")
parser.add_argument("--dropout", type=float, default=0.0, help="Amount of dropout per discriminator layer.")
parser.add_argument("--l2-weight", type=float, default=0.0, help="L2 regularization weight used (Torch calls it weight-decay)")
parser.add_argument("--lr", type=float, default=0.001, help="Yer good olde learning rate")


BATCH_SIZE = 64
FEATURE_SIZE = 60
# Hardcoded latent dim, pulled from the butt
# ("probably not as complex as speaker recognition")
LATENT_DIM = 256

# Terms:
#   filelist: A list of (speaker_id, filename, _, system_id, key)
# Note:
#   Assumes features are stored in DxN matrices, but inside code we will
#   handle them as NxD (preload_data transposes the data)


def create_network(args):
    # TODO hardcoded network layout. Change
    feature_network = XVectorLikeNetwork(
        feature_size=FEATURE_SIZE,
        output_size=LATENT_DIM,
        include_mean="mean" in args.stats,
        include_std="std" in args.stats,
        include_max="max" in args.stats,
        dropout=args.dropout
    ).cuda()

    model = SiameseWithDiscriminator(
        feature_size=LATENT_DIM,
        output_size=1,
        feature_network=feature_network,
        num_discriminator_layers=2,
        num_discriminator_units=256,
        discriminator_dropout=args.dropout
    ).cuda()

    return model, feature_network


def create_sample_batch_ce(bonafide_samples, spoof_samples, batch_size):
    """
    Return torch tensors ((input1, input2), target) for siamese
    network training (for mutual information with cross entropy).

    Constructs each batch to have roughly equal amount of target (1)
    and non-target (0) samples to keep training balanced.

    "Target" sample is when both samples are bonafide.
    "Non-target" is when one sample is bonafide and second is spoof.
    """
    input1_tensor_list = []
    input2_tensor_list = []
    target_tensor = torch.zeros(batch_size, 1).float()

    for i in range(batch_size):
        # Throw a coin if we add target or non-target sample
        input1 = None
        input2 = None
        target = None
        if random.random() < 0.5:
            # Target sample:
            #  Both samples are bonafide
            input1, input2 = random.sample(bonafide_samples, 2)
            target = 1
        else:
            # Non-target sample
            #  First is bonafide, second is spoof
            input1 = random.choice(bonafide_samples)
            input2 = random.choice(spoof_samples)
            target = 0

        # Put sampled vectors to batch
        input1_tensor_list.append(input1)
        input2_tensor_list.append(input2)
        target_tensor[i, 0] = target

    # TODO data augmentation? This was rather slow with ASV

    target_tensor = target_tensor.cuda()

    return ((input1_tensor_list, input2_tensor_list), target_tensor)


def create_sample_batch_contrastive(bonafide_samples, spoof_samples, batch_size):
    """
    Return torch tensors ((input1, input2), target) for siamese
    network training (for mutual information with Noise Contrastive
    Estimation. See https://arxiv.org/pdf/1812.00271.pdf ).

    Batch has one positive sample (first item input1/input2, bonafide
    again bonafide), while rest are negative ones (bonafide against spoof)
    """
    input1_tensor_list = []
    input2_tensor_list = []
    target_tensor = torch.zeros(batch_size, 1).float()

    # Sample random bonafide samples
    bonafide_sample, input2 = random.sample(bonafide_samples, 2)

    # Add this first sample
    input1_tensor_list.append(bonafide_sample)
    input2_tensor_list.append(input2)
    target_tensor[0, 0] = 1

    # Now gather batch_size - 1 negative samples, against the same
    # bonafide sample as sampled earlier.
    random_spoof_samples = random.sample(spoof_samples, batch_size - 1)
    input1_tensor_list.extend([bonafide_sample] * (batch_size - 1))
    input2_tensor_list.extend(random_spoof_samples)
    target_tensor[1:, 0] = 0

    return ((input1_tensor_list, input2_tensor_list), target_tensor)


def save_cm_model(feature_network, model, bonafide_samples, filename):
    """
    Save model to filename
    """
    # Use trained network to get average encoding for bonafide samples
    # Remember to set feature network to evaluation mode in case there are
    # dropouts
    feature_network.eval()
    bonafide_encodings = []
    for bonafide_sample in bonafide_samples:
        bonafide_encodings.append(feature_network(bonafide_sample[None])[0].cpu().detach().numpy())
    bonafide_encodings = np.array(bonafide_encodings)
    average_bonafide = np.mean(bonafide_encodings, axis=0)

    # Store the average_bonafide sample with the state dict
    state_dict = model.state_dict()
    state_dict["average_bonafide"] = average_bonafide

    torch.save(state_dict, filename)

    # Resume back to training setup for dropouts
    feature_network.train()


def main_train(args):
    model, feature_network = create_network(args)

    if args.feature_parameters is not None:
        # Preload parameters for the feature network
        state_dict = torch.load(args.feature_parameters)

        feature_network.load_state_dict(state_dict)

    trainable_parameters = None
    if args.freeze_feature:
        # Only update discriminator part feature_parameter
        trainable_parameters = model.discriminator.parameters()
    else:
        # Update whole (siamese/features + discriminator)
        trainable_parameters = model.parameters()

    optimizer = torch.optim.Adam(trainable_parameters, weight_decay=args.l2_weight, lr=args.lr)

    bonafide_samples, spoof_samples = load_cm_train_data(args.filelist, args.feature_directory)

    # Turn all samples into torch tensors and move to GPU already
    # (they are not that big)
    numpy_to_torch_cuda = lambda arr: torch.from_numpy(arr).float().cuda()
    bonafide_samples = list(map(numpy_to_torch_cuda, bonafide_samples))
    spoof_samples = list(map(numpy_to_torch_cuda, spoof_samples))

    num_samples = len(bonafide_samples) + len(spoof_samples)
    iterations = int(args.epochs * (num_samples // BATCH_SIZE))
    save_every_iterations = None
    if args.save_every_epochs:
        save_every_iterations = (args.save_every_epochs * (num_samples // BATCH_SIZE))

    progress_bar = tqdm(total=iterations)
    loss_deque = deque(maxlen=100)
    for update_i in range(iterations):
        inputs, targets = create_sample_batch_ce(
            bonafide_samples,
            spoof_samples,
            BATCH_SIZE
        )

        predictions = model(*inputs)

        # Cross-entropy loss
        predictions = torch.sigmoid(predictions)
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
            save_cm_model(feature_network, model, bonafide_samples, args.model + "_epochs_{}".format(epochs_passed))

    save_cm_model(feature_network, model, bonafide_samples, args.model)


def main_eval(args):
    """
    Evaluate trained model on given filelist
    """

    model, feature_network = create_network(args)

    # Load model
    state_dict = torch.load(args.model)
    # Remove the bonafide_average vector
    average_bonafide = state_dict.pop("average_bonafide")
    model.load_state_dict(state_dict)

    # Fix first input to average_bonafide sample
    model.fixed_x1_encoding = torch.from_numpy(average_bonafide[None]).float().cuda()
    model.eval()

    is_bonafide, samples, sample_systems = load_cm_trial_data(
        args.filelist, args.feature_directory
    )

    scores = []
    target_scores = []
    nontarget_scores = []

    for i in tqdm(range(len(is_bonafide))):
        if is_bonafide[i] is None:
            # Do not process this sample, just write Nones
            scores.append(None)
            continue

        score = model(None, torch.from_numpy(samples[i][None]).float().cuda()).item()
        scores.append(score)

        # Separate to target/nontarget scores for EER calculations
        if is_bonafide[i]:
            target_scores.append(score)
        else:
            nontarget_scores.append(score)

    # If we want to store results, save them here
    if args.output is not None:
        line_strings = []
        for line_items in zip(is_bonafide, scores, sample_systems):
            line_strings.append(" ".join(map(str, line_items)))
        out_str = "\n".join(line_strings)

        with open(args.output, "w") as f:
            f.write(out_str)

    # Display EER
    target_scores = np.array(target_scores)
    nontarget_scores = np.array(nontarget_scores)

    eer = compute_eer(target_scores, nontarget_scores)
    print("Total target scores:    {}".format(len(target_scores)))
    print("Total nontarget scores: {}".format(len(nontarget_scores)))
    print("EER:                    {}".format(eer))


def main_bulk_eval(args):
    """
    Evaluate bunch of different models on the same dataset
    """
    model, feature_network = create_network(args)

    is_bonafide, samples, sample_systems = load_cm_trial_data(
        args.filelist, args.feature_directory
    )

    # Check how long filenames we have so we can do prettier printing
    longest_filename = max(map(len, args.models))
    print_template = "{:<%d}  {:>6}" % (longest_filename)
    # Print header
    print(print_template.format("", "EER"))

    for model_path in args.models:
        # Load model
        state_dict = torch.load(model_path)
        average_bonafide = state_dict.pop("average_bonafide")
        model.load_state_dict(state_dict)
        model.fixed_x1_encoding = torch.from_numpy(average_bonafide[None]).float().cuda()
        model.eval()

        # Scoring
        target_scores = []
        nontarget_scores = []
        scores = []
        for i in tqdm(range(len(is_bonafide)), desc="score", leave=False):
            # Skip Nones (missing files)
            if samples[i] is None:
                continue

            score = model(None, torch.from_numpy(samples[i][None]).float().cuda()).item()
            scores.append(score)

            # Separate to target/nontarget scores for EER calculations
            if is_bonafide[i]:
                target_scores.append(score)
            else:
                nontarget_scores.append(score)

        # Display EER
        target_scores = np.array(target_scores)
        nontarget_scores = np.array(nontarget_scores)
        scores = np.array(scores)

        eer = compute_eer(target_scores, nontarget_scores)

        print(print_template.format(model_path, round(eer, 4)))

        # If we want to store individual scores
        if args.output is not None:
            line_strings = []
            for line_items in zip(is_bonafide, scores, sample_systems):
                line_strings.append(" ".join(map(str, line_items)))
            out_str = "\n".join(line_strings)

            filename = os.path.join(args.output, os.path.basename(model_path) + ".txt.cm_scores")
            with open(filename, "w") as f:
                f.write(out_str)


def main_extract_encodings(args):
    """
    Extract feature encodings of filelits 
    """
    assert args.output is not None, "Must provide --output for encode extraction"

    model, feature_network = create_network(args)

    # Load model
    state_dict = torch.load(args.model)
    # Remove the average_bonafide key
    _ = state_dict.pop("average_bonafide")
    model.load_state_dict(state_dict)

    model.eval()

    bonafide_samples, spoof_samples, spoof_systems = load_cm_trial_data(
        args.filelist, args.feature_directory, skip_missing=True, return_systems=True
    )

    bonafide_encodings = []
    spoof_encodings = []

    for bonafide_sample in tqdm(bonafide_samples):
        # x1 is fixed to be the average bonafide sample
        encoding = feature_network(torch.from_numpy(bonafide_sample[None]).float().cuda()).cpu().detach().numpy()
        # No need to sigmoid here
        bonafide_encodings.append(encoding[0])

    for spoof_sample in tqdm(spoof_samples):
        encoding = feature_network(torch.from_numpy(spoof_sample[None]).float().cuda()).cpu().detach().numpy()
        # No need to sigmoid here
        spoof_encodings.append(encoding[0])

    bonafide_encodings = np.array(bonafide_encodings)
    spoof_encodings = np.array(spoof_encodings)
    spoof_systems = np.array(spoof_systems)

    # Save results on disk
    np.savez(
        args.output,
        bonafide=bonafide_encodings,
        spoof=spoof_encodings,
        spoof_systems=spoof_systems
    )


if __name__ == "__main__":
    args = parser.parse_args()
    if args.operation == "train":
        main_train(args)
    elif args.operation == "eval":
        main_eval(args)
    elif args.operation == "bulk-eval":
        main_bulk_eval(args)
    else:
        main_extract_encodings(args)
