# Turn a list of .wav files into a format suitable for
# kaldi processing.
# Each utterance will be named according to the path listed in the list
import argparse
import sys
import os
import subprocess

parser = argparse.ArgumentParser("Turn list of WAV files into Kaldi files")
parser.add_argument("input_list", help="Path to file listing .wav files")
parser.add_argument("data_dir", type=str, help="Where wav files are stored (input_list has relative paths to this)")
parser.add_argument("output_dir", help="Path where to store Kaldi files (scp, utt2spk)")


def main(args):
    wav_files = open(args.input_list).readlines()
    wav_files = map(lambda x: x.strip(), wav_files)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    output_wav_file = os.path.join(args.output_dir, "wav.scp")
    output_utt_file = os.path.join(args.output_dir, "utt2spk")

    wavscp_lines = []
    utt_lines = []
    for wav_file in wav_files:
        utt_lines.append("{} {}".format(wav_file, wav_file))
        wavscp_lines.append("{} {}".format(wav_file, os.path.join(args.data_dir, wav_file)))

    print(wavscp_lines[0])
    print(utt_lines[0])

    with open(output_utt_file, "w") as utt2spk_file:
        utt2spk_file.write("\n".join(utt_lines))
    with open(output_wav_file, "w") as wavscp_file:
        wavscp_file.write("\n".join(wavscp_lines))

    result = subprocess.run(['utils/utt2spk_to_spk2utt.pl', os.path.join(args.output_dir, 'utt2spk')], stdout=open(os.path.join(args.output_dir, 'spk2utt'), 'w'), stderr=subprocess.STDOUT)
    result = subprocess.run(['utils/fix_data_dir.sh', args.output_dir], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)