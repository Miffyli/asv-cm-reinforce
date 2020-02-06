#!/bin/bash
# Kaldi recipe for extracting x-vectors from the ASVSpoof2019 dataset.
# Based on the kaldi/egs/voxceleb/v2 experiments with pretrained model from
#  https://kaldi-asr.org/models/m7  .

. ./cmd.sh
. ./path.sh
set -e

# Path to a text file listing all .wav files we want xvectors for
wav_list_location="[Point this to a .txt file listing all wav files we want x-vectors for]"
# Prefix for the paths in the above filelist (I.e. read final paths from wav_list_prefix + wav_path)
wav_list_prefix="[Prefix to add to path for each wav file listed in wav_list_location]"

# Where to store all the created data
datafolder="[path where to store all the data. Note that this will take quite a bit of space]"

# Experiment directory (under datafolder)
exp_dir="$datafolder/exp"

# Where different data should be stored under. Preferably
# under datafolder
mfccdir=$datafolder/mfcc
vaddir=$datafolder/mfcc
xvectordir=$datafolder/xvector

nnet_dir=exp/xvector_nnet_1a

stage=0

if [ $stage -le 0 ]; then  
  python3 scripts/make_kaldi_lists.py $wav_list_location $wav_list_prefix $datafolder
fi

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 8 --cmd "$train_cmd" \
    $datafolder $exp_dir/make_mfcc $mfccdir
  utils/fix_data_dir.sh $datafolder
  sid/compute_vad_decision.sh --nj 8 --cmd "$train_cmd" \
    $datafolder $exp_dir/make_vad $vaddir
  utils/fix_data_dir.sh $datafolder
fi


if [ $stage -le 9 ]; then
  # Extract x-vectors used in the evaluation.
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 8 \
  $nnet_dir $datafolder \
  $xvectordir
fi

