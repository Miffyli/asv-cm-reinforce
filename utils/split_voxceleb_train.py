# Create training list for VoxCeleb based on the
# trial list.
# Note: Assumes data is structured in directory in format
# speaker_name/"videoid_clipnumber.wav"
import argparse
import os

parser = argparse.ArgumentParser("Create list of training .wav files for VoXCeleb")
parser.add_argument("wav_list", help="List of all wav files in voxceleb folder")
parser.add_argument("trial_list", help="List of trials (one from voxceleb website)")
parser.add_argument("output", help="Where to store resulting list of training wavs")

# Trial list should contain lines
#   target_or_not input1 input2
# wav_list just contains paths to wav files
#   wav_file

def main(args):
    wav_list = open(args.wav_list).readlines()
    wav_list = list(map(lambda x: list(x.strip().split(" ")), wav_list))

    trial_list = open(args.trial_list).readlines()
    trial_list = list(map(lambda x: list(x.strip().split(" ")), trial_list))

    # Gather speakers in trials to remove them from
    # training wavs
    trial_speakers = set()
    for target_or_not, utterance1, utterance2 in trial_list:
        # Utterance 1/2 are "speaker_name/utteranceid_clipnumber.wav"
        speaker1 = utterance1.split("/")[0]
        speaker2 = utterance2.split("/")[0]
        trial_speakers.add(speaker1)
        trial_speakers.add(speaker2)

    # Now go over all wav files and remove ones that
    # have utterances from trial speakers
    train_wavs = []

    for (wav_path,) in wav_list:
        # Get the speaker name (the directory name in which file resides)
        speaker_name = os.path.basename(os.path.dirname(wav_path))
        if not speaker_name in trial_speakers:
            train_wavs.append(wav_path)

    # And store resulting list back to disk
    with open(args.output, "w") as f:
        f.write("\n".join(train_wavs))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
