# Go through a directory, gather all files with
# specific postfix and attempt converting them into
# .wav files with 16-bit, 16khz sampling rate.
import argparse
import os
import subprocess

from tqdm import tqdm

parser = argparse.ArgumentParser("Gather audio files from directory and turn them into .wav files")
parser.add_argument("directory", help="Directory to scan through")
parser.add_argument("output", help="Directory where converted samples should be placed to")
parser.add_argument("--postfix", type=str, default=".flac", help="Postfix of files to convert")

FFMPEG_PATH = "ffmpeg"
FFMPEG_TEMPLATE = FFMPEG_PATH + " -y -hide_banner -loglevel panic -i {input_file} -acodec pcm_s16le -ac 1 -ar 16000 {output_file}"


def convert_file(input_file, output_file):
    subprocess.check_output(
        FFMPEG_TEMPLATE.format(
            input_file=input_file,
            output_file=output_file
        ),
        shell=True,
    )


def main(args):
    progress_bar = tqdm()
    for dirpath, dirnames, filenames in os.walk(args.directory):
        for filename in filenames:
            if filename.endswith(args.postfix):
                original_path = os.path.join(dirpath, filename)
                target_path = original_path.replace(args.directory, args.output)
                target_path = target_path.replace(args.postfix, ".wav")

                # Make sure directories exist
                os.makedirs(os.path.dirname(target_path), exist_ok=True)

                convert_file(original_path, target_path)

                progress_bar.update(1)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)