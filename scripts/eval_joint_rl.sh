#!/bin/bash
# Evaluate the final joint-training models

# NOTE: Assumes that we already ran scoring and combined scores
for repetition in 1 2 3
do
  echo "--- Evaluating model ${repetition} ---"
  for dataset in "dev" "eval"
  do
    echo "--- Dataset ${dataset} ---"
    for experiment in "ce_same" "ce_split" "pg_simple" "pg_penalize" "pg_reward" "pg_tdcf"
    do
      echo "--- Experiment ${experiment} ---"
      python3 evaluate_tdcf.py \
        bulk_scores/asv_${dataset}/joint_rl_${experiment}_repetition_${repetition}_asv_model.txt.asv_scores \
        bulk_scores/cm_${dataset}/joint_rl_${experiment}_repetition_${repetition}_cm_model.txt.cm_scores
    done
  done
done
