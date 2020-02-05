#!/bin/bash
# Run bulk evaluation on asv

if test -z "$1"
then
    echo "Usage: bulk_score_cm_on_asvspoof path_to_model [path_to_model ...]"
    exit
fi

for dataset in "dev" "eval"
do
  for gender in "male" "female"
  do
    mkdir -p bulk_scores/cm_${dataset}_${gender}
    python3 train_siamese_cm.py \
        ASVSpoof2019_lists/ASVspoof2019.LA.asv.${dataset}.${gender}.trl.txt \
        features/cqcc/ASVspoof2019_LA_${dataset}/wav/ \
        none \
        bulk-eval \
        --output bulk_scores/cm_${dataset}_${gender} \
        --models ${@:1}
  done
done

