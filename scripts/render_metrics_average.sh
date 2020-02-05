#!/bin/bash
# Render score animations for an experiment

if test -z "$1"
then
    echo "Usage: render_score_animations_genderless experiment_name1 experiment_name2 ..."
    exit
fi

for dataset in "dev" "eval"
do
    python3 plot.py \
        average-metrics \
        bulk_scores/asv_${dataset}/ \
        bulk_scores/cm_${dataset}/ \
        output/${1}_${dataset}_averages.pdf \
        --experiment-names ${@:1}
    python3 plot.py \
        average-metrics \
        bulk_scores/asv_${dataset}/ \
        bulk_scores/cm_${dataset}/ \
        output/${1}_${dataset}_averages_relative.pdf \
        --relative \
        --experiment-names ${@:1}
done

