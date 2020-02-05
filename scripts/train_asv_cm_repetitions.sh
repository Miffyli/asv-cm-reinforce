#!/bin/bash
# Pre-train ASV and CM systems on the
# individual datasets (three times)

# Hyperparameters for ASV, obtained through
# grid-search
asv_epochs=60
asv_save_every_epochs=5
asv_l2_weight=0.00005
asv_dropout=0.0

# Train ASV models with three repetitions
for repetition in 1 2 3
do
  python3 train_xvector_asv.py \
    lists/VoxCeleb_asv_train_list.txt \
    features/xvectors/VoxCeleb/ \
    bulk_models/asv_repetition_${repetition} \
    train \
    --epochs $asv_epochs \
    --save-every-epochs $asv_save_every_epochs \
    --dropout $asv_dropout \
    --l2-weight $asv_l2_weight
  # Adapt final model to ASVSpoof
  scripts/adapt_asv_on_asvspoof.sh bulk_models/asv_repetition_${repetition}
done

cm_epochs=10
cm_save_every_epochs=5
cm_l2_weight=0.0
cm_dropout=0.0

# Train CM models with three repetitions
for repetition in 1 2 3
do
  python3 train_siamese_cm.py \
    ASVSpoof2019_lists/ASVspoof2019.LA.cm.train.trn.txt \
    features/cqcc/ASVspoof2019_LA_train/wav/ \
    bulk_models/cm_repetition_${repetition} \
    train \
    --epochs $cm_epochs \
    --save-every-epochs $cm_save_every_epochs \
    --dropout $cm_dropout \
    --l2-weight $cm_l2_weight
done
