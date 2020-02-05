#!/bin/bash
# Run joint RL and SL training with different reward
# functions on all three repetitions

# Fixed parameters
lr=0.0001
epochs=50
snapshotrate=250

for repetition in 1 2 3
do
  # REINFORCE Simple
  ./scripts/run_joint_rl_experiment.sh \
    joint_rl_pg_simple_repetition_${repetition} \
    bulk_models/asv_repetition_${repetition}_adapted_to_asvspoof \
    bulk_models/cm_repetition_${repetition} \
    --epochs ${epochs} \
    --snapshot-rate ${snapshotrate} \
    --reward-model simple \
    --lr ${lr}

  # REINFORCE Reward
  ./scripts/run_joint_rl_experiment.sh \
    joint_rl_pg_reward_repetition_${repetition} \
    bulk_models/asv_repetition_${repetition}_adapted_to_asvspoof \
    bulk_models/cm_repetition_${repetition} \
    --epochs ${epochs} \
    --snapshot-rate ${snapshotrate} \
    --reward-model reward \
    --lr ${lr}

  # REINFORCE Penalize
  ./scripts/run_joint_rl_experiment.sh \
    joint_rl_pg_penalize_repetition_${repetition} \
    bulk_models/asv_repetition_${repetition}_adapted_to_asvspoof \
    bulk_models/cm_repetition_${repetition} \
    --epochs ${epochs} \
    --snapshot-rate ${snapshotrate} \
    --reward-model penalize \
    --lr ${lr}

  # REINFORCE t-DCF
  # Also enable target/nontarget/spoof priors
  ./scripts/run_joint_rl_experiment.sh \
    joint_rl_pg_tdcf_repetition_${repetition} \
    bulk_models/asv_repetition_${repetition}_adapted_to_asvspoof \
    bulk_models/cm_repetition_${repetition} \
    --epochs ${epochs} \
    --snapshot-rate ${snapshotrate} \
    --reward-model tdcf \
    --priors \
    --lr ${lr}
  
  # Independent models, same labels
  ./scripts/run_joint_rl_experiment.sh \
    joint_rl_ce_same_repetition_${repetition} \
    bulk_models/asv_repetition_${repetition}_adapted_to_asvspoof \
    bulk_models/cm_repetition_${repetition} \
    --epochs ${epochs} \
    --snapshot-rate ${snapshotrate} \
    --loss ce \
    --lr ${lr}
  
  # Independent models, split labels
  ./scripts/run_joint_rl_experiment.sh \
    joint_rl_ce_split_repetition_${repetition} \
    bulk_models/asv_repetition_${repetition}_adapted_to_asvspoof \
    bulk_models/cm_repetition_${repetition} \
    --epochs ${epochs} \
    --snapshot-rate ${snapshotrate} \
    --loss ce_split \
    --lr ${lr}
done
