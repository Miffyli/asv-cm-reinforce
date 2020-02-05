# Take in Kaldi .scp (and optionally .ark) file,
# and extract vectors in the .ark to numpy format
# to file locations specified by speaker-id
# (Which, in this project, are relative paths to the dataset)
import numpy as np
import kaldi.util.io as kio
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser("Turn Kaldi scp/ark files to numpy files")
parser.add_argument("scp_file", help="Kaldi SCP file containing ARK info")
parser.add_argument("--ark-file", help="Path to ARK file. If given, override ARK file in SCP.")
parser.add_argument("output_directory", help="Prefix for the path where numpy files should be stored")


def main(args):
    scp_data = open(args.scp_file).readlines()
    scp_data = list(map(lambda x: list(x.strip().split(" ", 1)), scp_data))

    # If ark file is given as a parameter, use that file instead
    # of what is in the scp_data
    if args.ark_file is not None:
        # SCP file seems to have two // in the path info
        ark_file = args.ark_file.replace("/", "//")
        for utterance_info in scp_data:
            # Take the offset info (":1234") and use it
            # together with path to ark-file
            utterance_info[1] = "{}:{}".format(ark_file, utterance_info[1].split(":")[-1])

    for speaker_id, extended_filename in tqdm(scp_data):
        data = kio.read_vector(extended_filename).numpy()
        # Change file ending for clarity
        filepath = os.path.join(args.output_directory, speaker_id.replace(".wav", ".npy"))

        # Make sure directories exist
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

        np.save(filepath, data)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)