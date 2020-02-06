# Go through a directory, gather all .wav
# files and create a list of all .wav files under
# that directory
import argparse
import os

parser = argparse.ArgumentParser("Gather wav files from a directory")
parser.add_argument("directory", help="Directory to scan for wav files")
parser.add_argument("output", help="File where to store the wav files")


def main(args):
    wav_files = []
    for dirpath, dirnames, filenames in os.walk(args.directory):
        for filename in filenames:
            if ".wav" in filename:
                filepath = os.path.join(dirpath, filename)
                wav_files.append(os.path.relpath(filepath, args.directory))
    with open(args.output, "w") as f:
        f.write("\n".join(wav_files))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)