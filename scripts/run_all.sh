#!/bin/bash
# Run all main experiments from start to finish,
# _after_ all necessary features have been extracted

# Create output directories
mkdir -p bulk_models bulk_scores output
# Pretrain three ASV and CM models
./scripts/train_asv_cm_repetitions.sh
# Evaluate t-DCF and EERs of pretrained models
./scripts/eval_asv_cm_repetitions.sh > output/initial_eval.txt
# Joint/Tandem optimize the three repetitions
./scripts/run_joint_training.sh
./scripts/concatenate_gender_scores.sh
# Evaluate joint training
./scripts/eval_joint_rl.sh > output/joint_training_eval.txt
# Draw result plots
./scripts/render_metrics_average_repetitions.sh
./scripts/render_metrics_average_repetitions_paper.sh
