#!/bin/bash
# Evaluate the independent models
# on t-DCF 

# Run scoring (hardcoded list of models, derp)
./scripts/bulk_score_asv_on_asvspoof.sh \
    bulk_models/asv_repetition_1 \
    bulk_models/asv_repetition_2 \
    bulk_models/asv_repetition_3 \
    bulk_models/asv_repetition_1_adapted_to_asvspoof \
    bulk_models/asv_repetition_2_adapted_to_asvspoof \
    bulk_models/asv_repetition_3_adapted_to_asvspoof

./scripts/bulk_score_cm_on_asvspoof.sh \
    bulk_models/cm_repetition_1 \
    bulk_models/cm_repetition_2 \
    bulk_models/cm_repetition_3

# Concatenate scores for genderless scores
for repetition in 1 2 3
do
  for dataset in "dev" "eval"
  do
    cat bulk_scores/asv_${dataset}_male/asv_repetition_${repetition}.txt.asv_scores \
        bulk_scores/asv_${dataset}_female/asv_repetition_${repetition}.txt.asv_scores \
        > bulk_scores/asv_${dataset}/asv_repetition_${repetition}.txt.asv_scores

    cat bulk_scores/asv_${dataset}_male/asv_repetition_${repetition}_adapted_to_asvspoof.txt.asv_scores \
        bulk_scores/asv_${dataset}_female/asv_repetition_${repetition}_adapted_to_asvspoof.txt.asv_scores \
        > bulk_scores/asv_${dataset}/asv_repetition_${repetition}_adapted_to_asvspoof.txt.asv_scores

    cat bulk_scores/cm_${dataset}_male/cm_repetition_${repetition}.txt.cm_scores \
        bulk_scores/cm_${dataset}_female/cm_repetition_${repetition}.txt.cm_scores \
        > bulk_scores/cm_${dataset}/cm_repetition_${repetition}.txt.cm_scores
  done
done

# Compute scores
for repetition in 1 2 3
do
  echo "\n--- Evaluating model ${repetition} ---"
  for dataset in "dev" "eval"
  do
    echo "--- Dataset ${dataset} ---"
    echo "ASV with adaptation"
    python3 evaluate_tdcf.py \
        bulk_scores/asv_${dataset}/asv_repetition_${repetition}_adapted_to_asvspoof.txt.asv_scores \
        bulk_scores/cm_${dataset}/cm_repetition_${repetition}.txt.cm_scores
  done
done
