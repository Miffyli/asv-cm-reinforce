# Utilities for preloading data from different datasets into some
# common format
import os
from collections import namedtuple

import numpy as np
from tqdm import tqdm


# An object to hold information for a single trial.
Trial = namedtuple(
    "Trial",
    (
        "trial_features",   # Input features for the input utterance
        "claimed_identity", # Speaker ID of the claimed identity
        "test_features",    # OR provide features against which we test
        "is_target",         # If this trial is a target or not
        "origin"            # Origin of trial_features. Used in ASVSpoof for labeling spoof samples
    )
)


def readlines_and_split_spaces(filepath):
    """
    Read lines from filepath and then split each line by
    whitespaces.
    """
    filelist = open(filepath).readlines()
    filelist = list(map(lambda x: list(x.strip().split(" ")), filelist))

    return filelist


def load_cm_train_data(filelist_path, feature_dir, skip_missing=False, return_systems=False):
    """
    Preload features for spoof and bonafide samples, returning two lists
    of bonafide_samples and spoof samples (in this order).

    Parameters:
        filelist_path (str): Path to the filelist
        feature_dir (str): Path to the directory that contains the features
        skip_missing (bool): If true, missing trials that are missing files
        return_systems (bool): If true, returns third list which will contain
                               systems of each spoofing sample
    """
    filelist = readlines_and_split_spaces(filelist_path)

    # Check that this is supported filelist type (ASVSpoof list)
    if not (len(filelist[0]) == 5 and filelist[0][2]) == "-":
        raise RuntimeError("Unknown filetype, expected ASVSpoof2019 filelist.")

    spoof_samples = []
    bonafide_samples = []
    spoof_systems = []

    num_skipped = 0
    for speaker_id, filename, _, system_id, key in tqdm(filelist, desc="load"):
        filepath = os.path.join(feature_dir, filename) + ".npy"

        if not os.path.isfile(filepath):
            if skip_missing:
                # All ok, skip it
                num_skipped += 1
                continue
            else:
                raise RuntimeError("Feature file does not exist: {}".format(filepath))

        # Transpose data to be NxD rather than stored DxN
        data = np.load(filepath).astype(np.float32).T
        if key == "spoof":
            spoof_samples.append(data)
            if return_systems:
                spoof_systems.append(system_id)
        else:
            bonafide_samples.append(data)

    if num_skipped > 0:
        print("[WARNING] Skipped {} files".format(num_skipped))

    if return_systems:
        # Return systems used to generate spoof samples as well
        return bonafide_samples, spoof_samples, spoof_systems
    else:
        return bonafide_samples, spoof_samples


def load_cm_trial_data(filelist_path, feature_dir):
    """
    Preload features for bonafide and spoof samples from a trial list.

    Similar to load_cm_train_data, but this one keeps the original ordering.

    Note: If feature file is missing for a trial, all corresponding items are
          None in the lists below.

    Returns:
        is_bonafide (List): True or False, if corresponding utterance is
                            bonafide or not.
        input_features (List): List of input features, one per utterance
        spoof_systems (List): System where system originated. Copied from
                              ASVSpoof file
    """
    filelist = readlines_and_split_spaces(filelist_path)

    # Check that this is supported filelist type (ASVSpoof list)
    # Support both ASV and CM protocol files
    if not ((len(filelist[0]) == 5 and filelist[0][2] == "-") or len(filelist[0]) == 4):
        raise RuntimeError("Unknown filetype, expected ASVSpoof2019 filelist.")

    is_bonafides = []
    input_features = []
    spoof_systems = []

    num_skipped = 0
    filename = None
    system_id = None
    is_bonafide = None
    for row_items in tqdm(filelist, desc="load"):
        if len(row_items) == 5:
            # CM protocol file
            _, filename, _, system_id, key = row_items
            is_bonafide = key == "bonafide"
        else:
            # ASV protocol file (4 items per line)
            _, filename, system_id, key = row_items
            is_bonafide = key != "spoof"

        filepath = os.path.join(feature_dir, filename) + ".npy"

        if not os.path.isfile(filepath):
            # Missing file. Add Nones to lists
            is_bonafides.append(None)
            input_features.append(None)
            spoof_systems.append(None)
            continue

        # Transpose data to be NxD rather than stored DxN
        data = np.load(filepath).astype(np.float32).T
        spoof_systems.append(system_id)
        input_features.append(data)
        is_bonafides.append(is_bonafide)

    if num_skipped > 0:
        print("[WARNING] Skipped {} files".format(num_skipped))

    return is_bonafides, input_features, spoof_systems


def _load_asv_per_speaker_asvspoof(filelist, feature_dir):
    """
    Preload speaker features and group them by speaker id.

    This function is for ASVSpoof2019 logical access filelists
    """
    speaker_data = {}
    for speaker_id, filename, _, system_id, key in tqdm(filelist, desc="load"):
        # ASVSpoof filelists also include spoof samples
        if key == "spoof":
            continue
        filepath = os.path.join(feature_dir, filename) + ".npy"
        data = np.load(filepath).astype(np.float32)
        speaker_data[speaker_id] = speaker_data.get(speaker_id, []) + [data]

    return speaker_data


def _load_asv_per_speaker_voxceleb(filelist, feature_dir):
    """
    Preload speaker features and group them by speaker id.

    This function is for VoxCeleb dataset, where filelist contains
    list of all .wav files.
    Assumes each speaker has its files under their own directory,
    and we will use that directory name as speaker id. E.g. the
    directory looks something like
        .../speaker_a/00.wav
        .../speaker_a/01.wav
        .../speaker_a/02.wav
        .../speaker_b/00.wav
        .../speaker_b/01.wav
        ...
    """
    speaker_data = {}
    # filelist is list of lists, even if the sublist only has
    # one item
    for (filepath,) in tqdm(filelist, desc="load"):
        # Speaker id is the parent directory of the file we are going to read
        # We could use splitting here, but for OS-compatibility we go with
        # os.path functions
        speaker_id = os.path.basename(os.path.dirname(filepath))
        filepath = os.path.join(feature_dir, filepath).replace(".wav", ".npy")
        data = np.load(filepath).astype(np.float32)
        speaker_data[speaker_id] = speaker_data.get(speaker_id, []) + [data]

    return speaker_data


def load_asv_per_speaker(filelist_path, feature_dir):
    """
    Preload features for valid speakers in the filelist, grouped by speaker
    id.

    Parameters:
        filelist_path (str): Path to the filelist containing speakers to use for training
        feature_dire (str): Where the features are stored
    Returns:
        Dictionary mapping speaker-id to list of features for that speaker.
    """
    filelist = readlines_and_split_spaces(filelist_path)

    if (len(filelist[0]) == 5 and filelist[0][2]) == "-":
        # ASVSpoof2019 logical access file
        return _load_asv_per_speaker_asvspoof(filelist, feature_dir)
    elif len(filelist[0]) == 1:
        # A list of .wav files, used for VoxCeleb.
        return _load_asv_per_speaker_voxceleb(filelist, feature_dir)
    else:
        raise RuntimeError("Unknown filelist type (file {})".format(filelist_path))


def _load_enroll_data_asvspoof(filelist, feature_dir):
    """
    Preload data for speaker enrollment for ASVSpoof2019 .trn list.

    Each line in filelist is list of two items [speaker_id, comma_separated_list].
    """
    speaker_data = {}

    for speaker_id, list_of_files in tqdm(filelist, desc="load"):
        speaker_files = list_of_files.split(",")

        speaker_features = []
        for speaker_file in speaker_files:
            filepath = os.path.join(feature_dir, speaker_file) + ".npy"
            data = np.load(filepath).astype(np.float32)
            speaker_features.append(data)

        speaker_data[speaker_id] = speaker_features

    return speaker_data


def load_enroll_data(filelist_path, feature_dir):
    """
    Preload data for speaker enrollment. Returns a dictionary mapping
    speaker-ids to list of features to-be-used for enrollment

    Parameters:
        filelist_path (str): Path to the filelist containing speakers to enroll
        feature_dire (str): Where the features are stored
    Returns:
        Dictionary mapping speaker-id to list of features for that speaker.
    """

    filelist = readlines_and_split_spaces(filelist_path)

    if (len(filelist[0]) == 2 and "," in filelist[0][1]):
        # Two items per line: speaker_id and list_of_features (separated by comma)
        # ASVSpoof2019 logical access enroll file
        return _load_enroll_data_asvspoof(filelist, feature_dir)
    else:
        raise RuntimeError("Unknown filelist type (file {})".format(filelist_path))


def _load_trial_data_asvspoof(filelist, feature_dir, skip_spoof=True):
    """
    Preload data for speaker trials for ASVSpoof2019 .trl list.

    Each line in filelist is four items [claimed_identity, utterance, system, target].
    """

    trial_list = []

    for claimed_identity, utterance_file, system, is_target in tqdm(filelist, desc="load"):
        # Skip spoof samples
        if skip_spoof and system != "bonafide":
            continue

        is_target = is_target == "target"

        utterance_path = os.path.join(feature_dir, utterance_file) + ".npy"
        data = np.load(utterance_path).astype(np.float32)

        # If data has 2D, assume it is CQCC features we want to transpose
        if data.ndim == 2:
            data = data.T

        trial = Trial(
            trial_features=data,
            claimed_identity=claimed_identity,
            test_features=None,
            is_target=is_target,
            origin=system
        )

        trial_list.append(trial)

    return trial_list


def _load_trial_data_voxceleb(filelist, feature_dir):
    """
    Preload data for speaker trials for Voxceleb verification trial list.

    Each line in filelist is three items [target_or_not utterance1 utterance2].
    """

    trial_list = []

    for target_or_not, utterance1_file, utterance2_file in tqdm(filelist, desc="load"):
        is_target = target_or_not == "1"

        utterance1_path = os.path.join(feature_dir, utterance1_file).replace(".wav", ".npy")
        data1 = np.load(utterance1_path).astype(np.float32)

        utterance2_path = os.path.join(feature_dir, utterance2_file).replace(".wav", ".npy")
        data2 = np.load(utterance2_path).astype(np.float32)

        trial = Trial(
            trial_features=data1,
            claimed_identity=None,
            test_features=data2,
            is_target=is_target,
            origin=None
        )

        trial_list.append(trial)

    return trial_list


def load_trial_data(filelist_path, feature_dir, **kwargs):
    """
    Preload trials into a list of Trial elements (see top of data_loading.py).
    NOTE: Only load genuine trials (e.g. if there are spoofed samples, skip these)

    Parameters:
        filelist_path (str): Path to the filelist containing the trial information
        feature_dire (str): Where the features for these trials are stored
    Returns:
        List of Trial elements, each representing single trial.
    """
    filelist = readlines_and_split_spaces(filelist_path)


    if (len(filelist[0]) == 4 and ("spoof" in filelist[0] or "target" in filelist[0])):
        # Four items per line: speaker_id, utterance, system, target/nontarget/spoof
        # ASVSpoof2019 logical access trial list
        return _load_trial_data_asvspoof(filelist, feature_dir, **kwargs)
    elif len(filelist[0]) == 3 and filelist[0][0] in ("0", "1"):
        # Three items per line, and first is 0 or 1 (target or nontarget)
        return _load_trial_data_voxceleb(filelist, feature_dir)
    else:
        raise RuntimeError("Unknown filelist type (file {})".format(filelist_path))


def load_joint_train_data_asvspoof(filelist_path, asv_feature_dir, cm_feature_dir):
    """
    Preload data for joint ASV+CM training from asvspoof19 dataset.

    Parameters:
        filelist_path (str): Path to the ASVSpoof ASV .trl file
        asv_feature_dir (str): Where the features for ASV system are stored
        cm_feature_dir (str): Where the features for CM system are stored
    Returns:
        List of features for ASV system
        List of features for CM system
        List of speaker labels
        List of booleans if this trial should be passed (True if target speaker
            and bona fide sample)
        List of booleans if this trial has a spoof sample or not.
    """

    filelist = readlines_and_split_spaces(filelist_path)

    asv_features = []
    cm_features = []
    speaker_labels = []
    is_targets = []
    is_spoofs = []

    for speaker_id, filename, system_id, key in tqdm(filelist, desc="load"):
        # ASV data
        filepath = os.path.join(asv_feature_dir, filename) + ".npy"
        data = np.load(filepath).astype(np.float32)
        asv_features.append(data)

        # CM data
        filepath = os.path.join(cm_feature_dir, filename) + ".npy"
        # Remember to transpose the features...
        # Great design there, dummy!
        data = np.load(filepath).astype(np.float32).T
        cm_features.append(data)

        speaker_labels.append(speaker_id)

        # Target = target speaker sample and bona fide
        # Spoof samples have key = "spoof"
        is_targets.append(key == "target")
        # Also include information on if the trial is a spoof
        # sample or not
        is_spoofs.append(key == "spoof")

    return asv_features, cm_features, speaker_labels, is_targets, is_spoofs
