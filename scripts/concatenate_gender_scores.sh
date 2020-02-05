#!/bin/bash
# Concatenate scores from different genders into one

mkdir -p bulk_scores/asv_dev bulk_scores/asv_eval bulk_scores/cm_dev bulk_scores/cm_eval

./scripts/concatenate_scores.sh bulk_scores/asv_dev_male/ bulk_scores/asv_dev_female/ bulk_scores/asv_dev/
./scripts/concatenate_scores.sh bulk_scores/asv_eval_male/ bulk_scores/asv_eval_female/ bulk_scores/asv_eval/
./scripts/concatenate_scores.sh bulk_scores/cm_dev_male/ bulk_scores/cm_dev_female/ bulk_scores/cm_dev/
./scripts/concatenate_scores.sh bulk_scores/cm_eval_male/ bulk_scores/cm_eval_female/ bulk_scores/cm_eval/
