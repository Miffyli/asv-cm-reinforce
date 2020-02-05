# Entroll speakers in a given list by calculating
# mean vectors over them.
import argparse
import os

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser("Create enrollment vectors for speakers")
parser.add_argument("filelist", help="ASVSpoof2019 filelist of speakers. All speakers listed will be enrolled.")
parser.add_argument("feature_directory", help="Location where xvectors are stored.")
parser.add_argument("enroll_directory", help="Location where enrolled vectors should be stored.")

def main(args):
    # Gather unique speaker IDs
    filelist = open(args.filelist).readlines()
    filelist = list(map(lambda x: x.strip().split(" "), filelist))

    if not os.path.isdir(args.enroll_directory):
        os.makedirs(args.enroll_directory)

    speaker_id_list = list(map(lambda x: x[0], filelist))
    unique_speakers = list(set(speaker_id_list))

    # Iterate over speakers, gather their samples,
    # compute mean and store it into the enroll directory.
    for speaker_id in tqdm(unique_speakers):
        # Gather audio files for this speaker.
        # Skip spoof samples
        speaker_files = [
            utterance[1] for utterance in filelist if 
                (utterance[0] == speaker_id and utterance[-1] == "bonafide")
        ]
        datas = []
        for speaker_file in speaker_files:
            filepath = os.path.join(args.feature_directory, speaker_file) + ".npy"
            datas.append(np.load(filepath))
        datas = np.array(datas)
        mean_vec = datas.mean(axis=0)

        speaker_file = os.path.join(args.enroll_directory, speaker_id)
        np.save(speaker_file, mean_vec)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)