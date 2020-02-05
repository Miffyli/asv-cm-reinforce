#!/bin/bash
# Run bulk evaluation on asv

if test -z "$1"
then
    echo "Usage: bulk_score_asv_on_asvspoof path_to_model [path_to_model ...]"
    exit
fi

for dataset in "dev" "eval"
do
  for gender in "male" "female"
  do
    mkdir -p bulk_scores/asv_${dataset}_${gender}
    python3 train_xvector_asv.py \
        lists/ASVspoof2019.LA.asv.${dataset}.${gender}.trn.txt \
        features/xvectors/ASVspoof2019_LA_${dataset}/wav/ \
        none \
        bulk-eval \
        --trial-list lists/ASVspoof2019.LA.asv.${dataset}.${gender}.trl.txt \
        --output bulk_scores/asv_${dataset}_${gender} \
        --models ${@:1}
  done
done
