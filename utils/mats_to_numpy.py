# Recursively find all .mat files (MATLAB mat files)
# and replace them with a numpy file.
# NOTE: Assumes each .mat file only has one key
import numpy as np
from scipy.io import loadmat
import argparse
import os

parser = argparse.ArgumentParser("Walk through directories and turn .mat files into .np")
parser.add_argument("directory", help="Directory to walk through")
parser.add_argument("key", help="Key of the variable in Mat files")
parser.add_argument("--replace", help="Replace old files by removing them")

def main(args):
    for dirpath, dirnames, filenames in os.walk(args.directory):
        for filename in filenames:
            if ".mat" in filename:
                filepath = os.path.join(dirpath, filename)
                new_filepath = filepath.replace(".mat", ".npy")
                try:
                    mat_data = loadmat(filepath)
                except Exception as e:
                    print("[WARNING] Could not open {}: {}".format(filepath, e))
                    continue
                np_data = mat_data[args.key].astype(np.float)
                np.save(new_filepath, np_data)

                if args.replace:
                    os.unlink(filepath)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)