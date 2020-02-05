#!/bin/bash
# Run rl_joint training experiment, compute scores and render videos,
# all in one place.
if test -z "$1"
then
    echo "Usage: run_joint_rl_experiment experiment_name asv_model cm_model [parameters_to_train ...]"
    exit
fi

python3 train_joint_rl.py \
  lists/ASVspoof2019.LA.asv.dev.gi.trl.txt \
  features/xvectors/ASVspoof2019_LA_dev/wav/ \
  features/cqcc/ASVspoof2019_LA_dev/wav/ \
  ${2} \
  ${3} \
  train \
  --output bulk_models/${1} \
  ${@:4}

# Do scoring (include snapshots and final model)
./scripts/bulk_score_asv_on_asvspoof.sh bulk_models/${1}_updates_*_asv_model bulk_models/${1}_asv_model
./scripts/bulk_score_cm_on_asvspoof.sh bulk_models/${1}_updates_*_cm_model bulk_models/${1}_cm_model
# And finally render the videos
./scripts/render_score_animations.sh ${1}
