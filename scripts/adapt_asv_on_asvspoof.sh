#!/bin/bash
# Adapt (fine-tune) ASV model on ASVSpoof data

if test -z "$1"
then
    echo "Usage: adapt_asv_on_asvspoof path_to_model"
    exit
fi

# Parameters designed for adaptation
epochs=20
lr=0.0001
optimizer="adam"

python3 train_xvector_asv.py \
  ASVSpoof2019_lists/ASVspoof2019.LA.cm.train.trn.txt \
  features/xvectors/ASVspoof2019_LA_train/wav \
  ${1}_adapted_to_asvspoof \
  train \
  --load-model $1 \
  --epochs $epochs \
  --lr $lr \
  --optimizer $optimizer 