#!/bin/bash
# Render score animations for an experiment

if test -z "$1"
then
    echo "Usage: render_score_animations experiment_name"
    exit
fi

for dataset in "dev" "eval"
do
  for gender in "male" "female"
  do
    python3 plot.py \
        score-animation \
        --metrics \
        bulk_scores/asv_${dataset}_${gender}/ \
        bulk_scores/cm_${dataset}_${gender}/ \
        $1 \
        output/${1}_${dataset}_${gender}.mp4
  done
done

