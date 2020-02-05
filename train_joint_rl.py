# Train joint systems using reinforcement
# learning

import argparse
import os
import pickle
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.plot_utils import plot_score_distribution, plot_det, compute_eer
from utils.data_loading import load_joint_train_data_asvspoof, load_trial_data, load_enroll_data, load_asv_per_speaker
from train_siamese_cm import create_network as create_cm_network, save_cm_model
from train_xvector_asv import create_network as create_asv_network

parser = argparse.ArgumentParser("Joint train ASV+CM systems with reinforcement learning")
parser.add_argument("joint_filelist", help="Location of ASVSpoof2019 filelist (LA .trl) for joint training, or enrollment data")
parser.add_argument("joint_asv_directory", help="Location of the ASV features for joint training and evaluation")
parser.add_argument("joint_cm_directory", help="Location of the CM features for joint training and evaluation")
parser.add_argument("asv_model", help="Path to the pretrained ASV model")
parser.add_argument("cm_model", help="Path to the pretrained CM model")
parser.add_argument("operation", choices=["train"])
parser.add_argument("--joint-model", type=str, help="Path to joint parameters, for evaluation")
parser.add_argument("--trial-list", type=str, help="Trial list used for evaluation")
parser.add_argument("--output", type=str, help="Where to store trained models or scores")
parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
parser.add_argument("--save-every-epochs", type=int, default=None, help="How often to save snapshots of the model.")
parser.add_argument("--l2-weight", type=float, default=0.0, help="L2 regularization weight used (Torch calls it weight-decay)")
parser.add_argument("--debug", action="store_true", help="Enable debug prints")
parser.add_argument("--lr", type=float, default=0.001, help="Yer good olde learning rate for SGD")
parser.add_argument("--loss", type=str, choices=["pg", "ce", "ce_split", "cascaded_pg"], default="pg", help="Loss used to train the systems joinly")
parser.add_argument("--reward-model", type=str, choices=["simple", "fa10", "tdcf", "reward", "penalize"], default="simple", help="Rewards to use for reward-based training")
parser.add_argument("--deterministic", action="store_true", help="Take determinsitic (thresholded) actions instead of sampling.")
parser.add_argument("--snapshot-rate", type=int, default=None, help="How many updates between saving a snapshot of the model")
parser.add_argument("--from-scratch", action="store_true", help="Do not load model parameters and train from scratch.")
parser.add_argument("--clip-probability", action="store_true", help="Clip probabilities to avoid stretching them out too much.")
parser.add_argument("--priors", action="store_true", help="Include cost-model priors in the rewards.")
parser.add_argument("--standardize-rewards", action="store_true", help="Standardize rewards with mean/std normalization.")
parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="sgd", help="Optimizer to use.")

# CM configs
parser.add_argument("--stats", type=str, choices=["mean", "std", "max"], default=["mean"], nargs="+",
                    help="Statistics to include pooling over features.")
parser.add_argument("--dropout", type=float, default=0.0, help="Amount of dropout per discriminator layer.")

# ASV configs
parser.add_argument("--preprocessing", type=str, choices=["center", "lda"], nargs="*",
                    help="Preprocessing steps to run on data.")
parser.add_argument("--lda-dim", type=int, default=150, help="Target dimensionality for LDA. If None, do not use.")

BATCH_SIZE = 64
CQCC_FEATURE_SIZE = 60
XVECTOR_FEATURE_SIZE = 512
# Hardcoded latent dim, pulled from the butt
# ("probably not as complex as speaker recognition")
CM_LATENT_DIM = 256

# Hardcoded probability clipping.
# If probability is higher than this, zero-out the gradients
# so that it can not higher than this.
# Likewise, if probability is lower than 1 - this, then prevent
# gradients from going down.
# This is a dirty trick to prevent scores from exploding
# into far ends.
PROBABILITY_CLIP = 0.99

# Reward model object containing bit more descriptive
# names for different rewards and priors
# fr = false accept of non-target sample
# fr_spoof = false accept of spoof sample
RewardModel = namedtuple("RewardModel",
    ["ta", "tr", "fa", "fa_spoof", "fr", "p_spoof", "p_target", "p_nontarget"]
)

# Global default priors from ASVSpoof19
DEFAULT_P_SPOOF = 0.05
DEFAULT_P_TARGET = 0.95 * 0.99
DEFAULT_P_NONTARGET = 0.95 * 0.01

# Prior of selecting a target sample for training
# (target and non-spoof)
TARGET_SAMPLE_P = 0.5


# Different reward setups for:
#   true_accept, true_reject, false_accept, false_reject
REWARD_MODELS = {
    # Simple reward: Reward for correct, penalize for incorrect
    "simple": RewardModel(ta=1, tr=1, fa=-1, fa_spoof=-1, fr=-1,
                          p_spoof=DEFAULT_P_SPOOF, p_target=DEFAULT_P_TARGET,
                          p_nontarget=DEFAULT_P_NONTARGET),
    # Reward setup based on ASVSpoof19 t-DCF cost parameters:
    # Penalize false accepts more
    "fa10": RewardModel(ta=1, tr=1, fa=-10, fa_spoof=-10, fr=-1,
                        p_spoof=DEFAULT_P_SPOOF, p_target=DEFAULT_P_TARGET,
                        p_nontarget=DEFAULT_P_NONTARGET),
    # t-DCF cost parameters: No reward, penalize
    # false accepts more than false rejects.
    "tdcf": RewardModel(ta=0, tr=0, fa=-1, fa_spoof=-1, fr=-0.1,
                        p_spoof=DEFAULT_P_SPOOF, p_target=DEFAULT_P_TARGET,
                        p_nontarget=DEFAULT_P_NONTARGET),
    # Non-negative reward: Do not penalize, only reward
    "reward": RewardModel(ta=1, tr=1, fa=0, fa_spoof=-1, fr=0,
                          p_spoof=DEFAULT_P_SPOOF, p_target=DEFAULT_P_TARGET,
                          p_nontarget=DEFAULT_P_NONTARGET),
    # Non-positive reward: Do not reward, only penalize
    "penalize": RewardModel(ta=0, tr=0, fa=-1, fa_spoof=-1, fr=-1,
                            p_spoof=DEFAULT_P_SPOOF, p_target=DEFAULT_P_TARGET,
                            p_nontarget=DEFAULT_P_NONTARGET),
}


def create_joint_sample_batch(
    asv_features,
    cm_features,
    speaker_label_to_indeces,
    target_indeces,
    nontarget_indeces,
    is_spoofs,
    batch_size
):
    """
    Return torch tensors (
        (
            (
                input1_cm_features,
                input2_cm_features
            ),
            (
                input1_asv_features,
                input2_asv_features
            )
        ),
        targets,
        is_spoof,
    ), where targets till if corresponding trial should be passed
    (same speaker and both are bonafide samples).

    Constructs each batch to have roughly equal amount of target
    and non-target samples to keep training balanced
    """
    input1_cm_features = []
    input2_cm_features = []
    input1_asv_tensor = torch.zeros(batch_size, XVECTOR_FEATURE_SIZE).float()
    input2_asv_tensor = torch.zeros(batch_size, XVECTOR_FEATURE_SIZE).float()
    target_tensor = torch.zeros(batch_size, 1).float()
    is_spoof_tensor = torch.zeros(batch_size, 1).float()

    speaker_labels = list(speaker_label_to_indeces.keys())

    for i in range(batch_size):
        # Throw a coin if we add target or non-target sample
        input1_idx = None
        input2_idx = None

        target = None
        is_spoof = None
        if random.random() < TARGET_SAMPLE_P:
            # Target sample:
            #   Pick speaker, and then pick two bona fide
            #   features from it
            random_speaker_label = random.choice(speaker_labels)
            # speaker_label_to_indeces only has indeces of target
            # samples that are also bona fide
            speaker_indices = speaker_label_to_indeces[random_speaker_label]
            # random.sample takes two unique vectors
            input1_idx, input2_idx = random.sample(speaker_indices, 2)
            target = 1
            is_spoof = 0
        else:
            # Non-target sample
            #  Three possibilities:
            #    1) Samples are bona fide but not target speaker
            #    2) At least one of the samples is not bona fide
            #    3) Parts 1 and 2 combined (non-target, spoofs)
            # nontarget_indeces array tells us which trials should be
            # failed (input is either bona-fide or non-target, but
            # we do not know which).
            # In realistic scenarios at least one of the samples
            # would be a legit sample (bona fide and of target speaker).
            # -> Pick one bad sample and then one of the target
            #    samples (any).
            input1_idx = random.choice(nontarget_indeces)
            input2_idx = random.choice(target_indeces)
            target = 0
            is_spoof = 0

            # Check if nontarget is spoof or not
            if is_spoofs[input1_idx]:
                is_spoof = 1

            # To "symmetrise" discriminator network,
            # flip inputs every now and then. Otherwise
            # one of the inputs will always be a legit
            # sample etc
            if random.random() < 0.5:
                temp_idx = input1_idx
                input1_idx = input2_idx
                input2_idx = temp_idx

        # Put sampled vectors to batch
        input1_asv_tensor[i] = torch.from_numpy(asv_features[input1_idx]).float().cuda()
        input2_asv_tensor[i] = torch.from_numpy(asv_features[input2_idx]).float().cuda()
        # These will be turned into Torch tensors later
        input1_cm_features.append(cm_features[input1_idx])
        input2_cm_features.append(cm_features[input2_idx])
        target_tensor[i, 0] = target
        is_spoof_tensor[i, 0] = is_spoof

    input1_asv_tensor = input1_asv_tensor.cuda()
    input2_asv_tensor = input2_asv_tensor.cuda()
    target_tensor = target_tensor.cuda()
    is_spoof_tensor = is_spoof_tensor.cuda()

    return (
        (
            (
                input1_cm_features,
                input2_cm_features
            ), (
                input1_asv_tensor,
                input2_asv_tensor
            )
        ),
        target_tensor,
        is_spoof_tensor
    )


def compute_pg_loss(asv_predictions, cm_predictions, targets, is_spoof, args):
    """
    Compute policy-gradient loss, which minimized should
    lead to better joint accuracy with asv and cm.

    Parameters:
        asv_predictions: Torch tensor of (N, 1), the single value representing
                         probability of passing this trial through ASV
        cm_predictions: Torch tensor of (N, 1), the single representing
                        probability of passing this trial through CM
        targets: Torch tensor of (N, 1), indicating if corresponding trial
                 is truly accept or reject
        args: Argparse arguments

    Return:
        Torch tensor with the loss to be minimized
    """

    # Sample actions
    asv_actions = None
    cm_actions = None

    if args.deterministic:
        # Threshold at 0.5
        asv_actions = asv_predictions > 0.5
        cm_actions = cm_predictions > 0.5
    else:
        # Stochastic actions
        asv_actions = torch.rand_like(asv_predictions) < asv_predictions
        cm_actions = torch.rand_like(cm_predictions) < cm_predictions

    # Only pass user when both are good with passing the user
    joint_action = asv_actions & cm_actions

    # Check if actions were TA/TR/FA/FR, and reward accordingly
    reward_model = REWARD_MODELS[args.reward_model]
    targets = targets.bool()
    is_spoof = is_spoof.bool()
    rewards = torch.zeros_like(targets).float()

    # True accept (both are one)
    rewards[joint_action & targets] = reward_model.ta
    # True reject (both are zero)
    rewards[(~joint_action) & (~targets)] = reward_model.tr
    # False accept (action is true but target is zero), non-targets
    rewards[((joint_action & (~targets)) & (~is_spoof))] = reward_model.fa
    # False accept (action is true but target is zero), spoofed samples
    rewards[((joint_action & (~targets)) & is_spoof)] = reward_model.fa_spoof
    # False reject (action is neg but target is one)
    rewards[(~joint_action) & targets] = reward_model.fr

    # Apply priors
    if args.priors:
        rewards[targets] *= reward_model.p_target
        rewards[(~targets) & (~is_spoof)] *= reward_model.p_nontarget
        rewards[is_spoof] *= reward_model.p_spoof

    # Compute PG loss:
    #   E[log pi * R]
    # We only have one step in our "MDP", so no need to
    # discount or anything here.
    # Assumption: ASV and CM actions are independent

    # Case 1: joint_action is true
    #   Joint probability is asv_prediction * cm_prediction,
    #   as these values are directly "probability of selecting true"
    # Case 2: joint_action is false
    #   Either asv or cm action was false or both were false.
    #   => not_asv * cm + asv * not_cm + not_asv * not_cm

    # First calculate Case 2 for all and then replace
    # correct parts with Case 1 calculation
    joint_action_p = (
        (1 - asv_predictions) * cm_predictions +
        asv_predictions * (1 - cm_predictions) +
        (1 - asv_predictions) * (1 - cm_predictions)
    )
    # Replace with Case 1
    joint_action_p[joint_action] = (
        (asv_predictions * cm_predictions)[joint_action]
    )

    # Clip probabilities
    if args.clip_probability:
        # Where rewards are positive, clip from above
        high_p_idx = (joint_action_p > PROBABILITY_CLIP) & (rewards > 0)
        low_p_idx = (joint_action_p < (1 - PROBABILITY_CLIP)) & (rewards < 0)
        joint_action_p[high_p_idx] = PROBABILITY_CLIP
        joint_action_p[low_p_idx] = (1 - PROBABILITY_CLIP)

    # Standardize rewards
    if args.standardize_rewards:
        rewards = (rewards - torch.mean(rewards)) / torch.std(rewards)

    # PG objective. Note that this is supposed to be maximized
    pg_objective = torch.log(joint_action_p + 1e-5) * rewards
    # Optimizer outside this function will try to minimized
    # returned value. Maximizing objective == minimizing
    # negative of it.
    pg_loss = -pg_objective

    # Mean over batch
    pg_loss = torch.mean(pg_loss)

    return pg_loss


def compute_pg_cascaded_loss(asv_predictions, cm_predictions, targets, is_spoof, args):
    """
    Compute policy gradient of a cascaded system where
    CM comes first and then ASV.

    Parameters:
        asv_predictions: Torch tensor of (N, 1), the single value representing
                         probability of passing this trial through ASV
        cm_predictions: Torch tensor of (N, 1), the single representing
                        probability of passing this trial through CM
        targets: Torch tensor of (N, 1), indicating if corresponding trial
                 is truly accept or reject
        args: Argparse arguments

    Return:
        Torch tensor with the loss to be minimized
    """

    # Sample actions
    if args.deterministic:
        # Threshold at 0.5
        asv_actions = asv_predictions > 0.5
        cm_actions = cm_predictions > 0.5
    else:
        # Stochastic actions
        asv_actions = torch.rand_like(asv_predictions) < asv_predictions
        cm_actions = torch.rand_like(cm_predictions) < cm_predictions

    # Joint action of a cascaded system (CM first, then ASV)
    #   - If CM is false, joint_action is False (regardless of ASV)
    #   - If CM is true but asv is False, joint_action is False
    #   - If both are true, then joint_action is True
    # Aaand this out to be same as in "parallel" systems.
    # However the difference is below when calculating PI.
    joint_action = asv_actions & cm_actions

    # Check if actions were TA/TR/FA/FR, and reward accordingly
    reward_model = REWARD_MODELS[args.reward_model]
    targets = targets.bool()
    is_spoof = is_spoof.bool()
    rewards = torch.zeros_like(targets).float()

    # True accept (both are one)
    rewards[joint_action & targets] = reward_model.ta
    # True reject (both are zero)
    rewards[(~joint_action) & (~targets)] = reward_model.tr
    # False accept (action is true but target is zero), non-targets
    rewards[((joint_action & (~targets)) & (~is_spoof))] = reward_model.fa
    # False accept (action is true but target is zero), spoofed samples
    rewards[((joint_action & (~targets)) & is_spoof)] = reward_model.fa_spoof
    # False reject (action is neg but target is one)
    rewards[(~joint_action) & targets] = reward_model.fr

    # Apply prior
    if args.priors:
        rewards[targets] *= reward_model.p_target
        rewards[(~targets) & (~is_spoof)] *= reward_model.p_nontarget
        rewards[is_spoof] *= reward_model.p_spoof

    if args.reward_scale is not None:
        rewards *= args.reward_scale

    # Calculate probabilities of taking these actions
    # Difference to compute_pg_loss is here:
    #   If CM action is false, we never query ASV, and thus
    #   nothing is backpropagated there.
    #   -> zero out asv_predictions for CM actions with false
    asv_predictions = asv_predictions * cm_actions.float()

    joint_action_p = (
        (1 - asv_predictions) * cm_predictions +
        asv_predictions * (1 - cm_predictions) +
        (1 - asv_predictions) * (1 - cm_predictions)
    )
    # Replace with Case 1
    joint_action_p[joint_action] = (
        (asv_predictions * cm_predictions)[joint_action]
    )

    # Clip probabilities
    if args.clip_probability:
        # Where rewards are positive, clip from above
        high_p_idx = (joint_action_p > PROBABILITY_CLIP) & (rewards > 0)
        low_p_idx = (joint_action_p < (1 - PROBABILITY_CLIP)) & (rewards < 0)
        joint_action_p[high_p_idx] = PROBABILITY_CLIP
        joint_action_p[low_p_idx] = (1 - PROBABILITY_CLIP)

    # Standardize rewards
    if args.standardize_rewards:
        rewards = (rewards - torch.mean(rewards)) / torch.std(rewards)

    # PG objective. Note that this is supposed to be maximized
    pg_objective = torch.log(joint_action_p + 1e-7) * rewards
    # Optimizer outside this function will try to minimized
    # returned value. Maximizing objective == minimizing
    # negative of it.
    pg_loss = -pg_objective

    # Mean over batch
    pg_loss = torch.mean(pg_loss)

    return pg_loss


def compute_ce_loss(asv_predictions, cm_predictions, targets, is_spoof, args):
    """
    Yer standard cross-entropy loss: Just try to get asv and cm predictions
    individually towards the targets.

    NOTE: This is not quite right, as the target label does not tell separately
          should asv pass and/or should cm pass. I.e. we end up training our
          ASV/CM systems on negative labels that are not negative for that subtask.

    Parameters:
        asv_predictions: Torch tensor of (N, 1), the single value representing
                         probability of passing this trial through ASV
        cm_predictions: Torch tensor of (N, 1), the single representing
                        probability of passing this trial through CM
        targets: Torch tensor of (N, 1), indicating if corresponding trial
                 is truly accept or reject
        args: Argparse arguments

    Return:
        Torch tensor with the loss to be minimized
    """

    asv_loss = torch.nn.functional.binary_cross_entropy(asv_predictions, targets)
    cm_loss = torch.nn.functional.binary_cross_entropy(cm_predictions, targets)

    total_loss = asv_loss + cm_loss
    return total_loss


def compute_ce_split_loss(asv_predictions, cm_predictions, targets, is_spoof, args):
    """
    Yer standard cross-entropy loss, but this time targets are split such that
    ASV and CM will still do their own tasks.

    Parameters:
        asv_predictions: Torch tensor of (N, 1), the single value representing
                         probability of passing this trial through ASV
        cm_predictions: Torch tensor of (N, 1), the single representing
                        probability of passing this trial through CM
        targets: Torch tensor of (N, 1), indicating if corresponding trial
                 is truly accept or reject
        args: Argparse arguments

    Return:
        Torch tensor with the loss to be minimized
    """

    is_spoof_bool = is_spoof.bool()

    # ASV only considers non-spoof samples
    asv_loss = torch.nn.functional.binary_cross_entropy(
        asv_predictions[~is_spoof_bool],
        targets[~is_spoof_bool]
    )

    is_spoof_float = is_spoof.float()
    # CM considers all samples, but target is 1.0 - is_spoof
    # (prediction 1 means accept)
    cm_loss = torch.nn.functional.binary_cross_entropy(
        cm_predictions,
        1.0 - is_spoof_float
    )

    total_loss = asv_loss + cm_loss
    return total_loss


# Mapping from simple name to the corresponding
# loss function
LOSS_FUNCTIONS = {
    "pg": compute_pg_loss,
    "ce": compute_ce_loss,
    "ce_split": compute_ce_split_loss,
    "cascaded_pg": compute_pg_cascaded_loss
}


def save_models(
    output_path,
    asv_model,
    asv_preprocessing_parameters,
    cm_feature_network,
    cm_model,
    bonafide_cm_features
):
    """
    Save current asv and cm model to output_path
    """
    asv_state_dict = asv_model.state_dict()
    # Add preprocessing data for Xvectors (if any)
    asv_state_dict.update(asv_preprocessing_parameters)
    torch.save(asv_state_dict, output_path + "_asv_model")

    # Use existing function to save CM model
    save_cm_model(
        cm_feature_network,
        cm_model,
        bonafide_cm_features,
        output_path + "_cm_model"
    )


def main_train(args):
    """
    Take pretrained ASV and CM and jointly
    train them on joint ASV+CM task using
    reinforcement learning or similar
    techniques (the two systems
    still stay independent of each other)
    """
    if args.preprocessing is not None:
        raise NotImplementedError("ASV Preprocessing is not yet supported")
    if args.output is None and args.snapshot_rate is not None:
        raise RuntimeError("Can not save snapshots without output")

    # Create and load cm network
    cm_model, cm_feature_network = create_cm_network(args)
    cm_state_dict = torch.load(args.cm_model)
    # Remove the bonafide_average vector stored with
    # CM models
    average_bonafide = cm_state_dict.pop("average_bonafide")
    # Do not load parameters if so desired
    if not args.from_scratch:
        cm_model.load_state_dict(cm_state_dict)
    # Get CM embedding size we are about to feed to ASV system
    cm_embedding_size = average_bonafide.shape[0]

    # Create and load ASV network
    asv_model, asv_feature_network = create_asv_network(XVECTOR_FEATURE_SIZE, args)
    asv_state_dict = torch.load(args.asv_model)
    # Remove any preprocessing steps
    preprocessing_parameters = {
        "centering_mean": asv_state_dict.pop("centering_mean", None),
        "lda": asv_state_dict.pop("lda", None)
    }
    if not args.from_scratch:
        asv_model.load_state_dict(asv_state_dict)

    asv_features, cm_features, speaker_labels, is_targets, is_spoofs = load_joint_train_data_asvspoof(
        args.joint_filelist, args.joint_asv_directory, args.joint_cm_directory,
    )

    # Move CM features to cuda already
    numpy_to_torch_cuda = lambda arr: torch.from_numpy(arr).float().cuda()
    cm_features = list(map(numpy_to_torch_cuda, cm_features))

    # Split bonafide spoof samples to a separate list for
    # saving CM model
    bonafide_cm_features = [cm_features[i] for i in range(len(cm_features)) if not is_spoofs[i]]

    # Preprocess data a little bit: Create a dictionary
    # that maps speaker labels to indeces where speaker has
    # bona fide samples
    speaker_label_to_indeces = {}
    for i, (speaker_label, is_target) in enumerate(zip(speaker_labels, is_targets)):
        if is_target:
            # Speaker label and input features match,
            # add this index to list of valid samples for that
            # speaker
            speaker_label_to_indeces[speaker_label] = (
                speaker_label_to_indeces.get(speaker_label, []) + [i]
            )

    # Also separate indeces of nontarget/target trials
    is_targets = np.array(is_targets)
    target_indeces = np.where(is_targets)[0]
    nontarget_indeces = np.where(~is_targets)[0]

    all_parameters = list(asv_model.parameters())
    all_parameters.extend(list(cm_model.parameters()))
    # SGD is stabler than Adam, but need to see how many epochs it takes to train
    optimizer = None
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(all_parameters, weight_decay=args.l2_weight, lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(all_parameters, weight_decay=args.l2_weight, lr=args.lr)

    num_samples = len(asv_features)
    total_iterations = int(args.epochs * (num_samples // BATCH_SIZE))

    progress_bar = tqdm(total=total_iterations)
    loss_deque = deque(maxlen=100)

    loss_function = LOSS_FUNCTIONS[args.loss]

    for update_i in range(total_iterations):
        (cm_inputs, asv_inputs), targets, is_spoof = create_joint_sample_batch(
            asv_features,
            cm_features,
            speaker_label_to_indeces,
            target_indeces,
            nontarget_indeces,
            is_spoofs,
            BATCH_SIZE
        )

        asv_predictions = asv_model(*asv_inputs)
        cm_predictions = cm_model(*cm_inputs)

        # Predictions are logits, turn into probablities
        asv_predictions = torch.sigmoid(asv_predictions)
        cm_predictions = torch.sigmoid(cm_predictions)

        # Loss should return a scalar value we then backprop through
        # and update parameters to minimize it
        loss = loss_function(asv_predictions, cm_predictions, targets, is_spoof, args)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_deque.append(loss.item())
        average_loss = np.mean(loss_deque)
        progress_bar.update(1)
        progress_bar.set_description("Loss: {:2.5f}".format(average_loss))
        progress_bar.refresh()

        if args.snapshot_rate is not None and (update_i % args.snapshot_rate) == 0:
            save_models(
                args.output + "_updates_{}".format(update_i),
                asv_model,
                preprocessing_parameters,
                cm_feature_network,
                cm_model,
                bonafide_cm_features
            )


    if args.output is not None:
        save_models(
            args.output,
            asv_model,
            preprocessing_parameters,
            cm_feature_network,
            cm_model,
            bonafide_cm_features
        )


if __name__ == "__main__":
    args = parser.parse_args()
    if args.operation == "train":
        main_train(args)
