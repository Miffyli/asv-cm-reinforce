# Implementation of t-DCF as described in
# "t-DCF: a Detection Cost Function for the Tandem Assessment of
#  Spoofing Countermeasures and Automatic Speaker Verification"
# By Tomi Kinnunen et al.
#
import argparse
import os
from collections import namedtuple

import numpy as np

# Notes:
#   - Terms "Positive" and "Acceptance" mean the same
#   - Terms "Negative" and "Reject" mean the same

CostParameters = namedtuple(
    "CostParameters",
    (
        "p_spoof",
        "p_tar",
        "p_nontar",
        "c_asv_miss",
        "c_asv_fa",
        "c_cm_miss",
        "c_cm_fa"
    )
)

# Cost-model from the Python tDCF code shared with
# ASVSpoof2019 competition
ASVSPOOF2019_COST_MODEL = CostParameters(
    p_spoof=0.05,
    p_tar=0.95 * 0.99,
    p_nontar=0.95 * 0.01,
    c_asv_miss=1,
    c_asv_fa=10,
    c_cm_miss=1,
    c_cm_fa=10
)


def compute_det(target_scores, nontarget_scores):
    """
    Compute DET curve (similar to ROC curve) over given target and
    nontarget scores. Assume target_scores should be high.

    This code is originally from ASVSpoof2019 code for t-DCF from here:
        https://www.asvspoof.org/

    Parameters:
        target_scores (ndarray): Array of scores for target trials (should be high)
        nontarget_scores (ndarray): Array of scores for nontarget trials (should be low)
    Returns
        frr (ndarray): False Rejection Rates at different thresholds
        far (ndarray): False Acceptance Rates at different thresholds
        thresholds (ndarray): Thresholds for previously mentioned points
    """
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds


def compute_eer(frr, far, thresholds):
    """
    Compute Equal Error Rate (EER) for given False rejection/acceptance rates.
    EER is when both FRR and FAR are the same. It is unlikely we have exactly
    same error rates in our FRR/FAR arrays, so we pick the one where these two
    are almost the same.

    This code is almost the same as in ASVSpoof2019 t-DCF code:
        https://www.asvspoof.org/

    Parameters:
        frr (ndarray): False rejection rates at different thresholds
        far (ndarray): False acceptance rates at different thresholds
        thresholds (ndarray): Thresholds for the above fars/frrs
    Returns:
        frr_eer (float): FRR near EER point
        far_eer (float): FAR near EER point
        eer_threshold (float): Threshold for the equal error rate
    """
    differences = np.abs(frr - far)
    # Find the spot where difference is smallest
    min_diff_idx = np.argmin(differences)
    # Our equal error rate is average frr/far at this point
    frr_eer = frr[min_diff_idx]
    far_eer = far[min_diff_idx]
    eer_threshold = thresholds[min_diff_idx]
    return frr_eer, far_eer, eer_threshold


def compute_eer_and_det(target_scores, nontarget_scores):
    """
    Compute DET curve and corresponding EER (see compute_eer and compute_det)

    Parameters:
        target_scores (ndarray): Array of scores for target trials (should be high)
        nontarget_scores (ndarray): Array of scores for nontarget trials (should be low)
    Returns
        frr_eer (float): FRR near approximate EER point
        far_eer (float): FAR near approximate EER point
        eer_threshold (float): Threshold for the equal error rate
        frr (ndarray): False Rejection Rates at different thresholds
        far (ndarray): False Acceptance Rates at different thresholds
        thresholds (ndarray): Thresholds for previously mentioned points
    """
    frr, far, thresholds = compute_det(target_scores, nontarget_scores)
    frr_eer, far_eer, eer_threshold = compute_eer(frr, far, thresholds)
    return frr_eer, far_eer, eer_threshold, frr, far, thresholds


def compute_asvspoof_tDCF(
    asv_target_scores,
    asv_nontarget_scores,
    asv_spoof_scores,
    cm_bonafide_scores,
    cm_spoof_scores,
    cost_model,
):
    """
    Compute t-DCF curve as in ASVSpoof2019 competition: 
        Fix ASV threshold to EER point and compute t-DCF curve over thresholds in CM.
    
    Code for this is mainly taken from the ASVSpoof2019 competition t-DCF implementation:
        https://www.asvspoof.org/

    Parameters:
        asv_target_scores (ndarray): Array of ASV target (bonafide) scores (should be high)
        asv_nontarget_scores (ndarray): Array of ASV nontarget (bonafide) scores (should be low)
        asv_spoof_scores (ndarray): Array of ASV spoof scores (should be low)
        cm_bonafide_scores (ndarray): Array of CM target (bonafide) scores (should be high)
        cm_spoof_scores (ndarray): Array of CM nontarget (spoof) scores (should be low)
        cost_model (CostParameters): CostParameters object containing cost parameters
    Returns: 
        tdcf_curve (ndarray): Array of normalized t-DCF values at different CM thresholds
        cm_thresholds (ndarray): Array of different CM thresholds, corresponding to
                                 values in tdcf_curve.
    """

    # Fix ASV FAR and miss to values at EER (with legit samples)
    asv_frr, asv_far, asv_thresholds = compute_det(asv_target_scores, asv_nontarget_scores)
    asv_frr_eer, asv_far_eer, asv_eer_threshold = compute_eer(asv_frr, asv_far, asv_thresholds)
    p_asv_miss = asv_frr_eer
    p_asv_fa = asv_far_eer

    # Fraction of spoof samples that were rejected by asv.
    # Note that speaker labels are not used here, just raw number
    # of spoof samples rejected by asv in general
    p_asv_spoof_miss = np.sum(asv_spoof_scores < asv_eer_threshold) / len(asv_spoof_scores)

    # Copy/pasta from t-DCF implementation in ASVSpoof2019 competition
    # Obtain miss and false alarm rates of CM
    p_cm_miss, p_cm_fa, cm_thresholds = compute_det(cm_bonafide_scores, cm_spoof_scores)

    # See ASVSpoof2019 evaluation plan for more information on these
    C1 = cost_model.p_tar * (cost_model.c_cm_miss - cost_model.c_asv_miss * p_asv_miss) - \
         cost_model.p_nontar * cost_model.c_asv_fa * p_asv_fa
    # Cost for CM false-accept: 
    #   How often we have spoof samples * 
    #   Cost of accepting a spoof * 
    #   how often ASV accepts spoof
    C2 = cost_model.c_cm_fa * cost_model.p_spoof * (1 - p_asv_spoof_miss)

    # Obtain t-DCF curve for all thresholds
    tDCF = C1 * p_cm_miss + C2 * p_cm_fa

    # Normalized t-DCF
    tDCF_norm = tDCF
    if min(C1, C2) == 0:
        tDCF_norm = tDCF
    else:
        tDCF_norm = tDCF / np.minimum(C1, C2)

    return tDCF_norm, cm_thresholds


if __name__ == "__main__":
    # Test if the above functions run (but does not check the logic)
    asv_target_scores = np.random.normal(4.0, 0.1, size=(500,))
    asv_nontarget_scores = np.random.normal(0.0, 2.0, size=(500,))
    cm_target_scores = np.random.normal(4.0, 0.1, size=(500,))
    cm_nontarget_scores = np.random.normal(1.0, 1.0, size=(500,))
    cost_model = ASVSPOOF2019_COST_MODEL

    frr, far, thresholds = compute_det(asv_target_scores, asv_nontarget_scores)
    frr_eer, far_eer, eer_threshold = compute_eer(frr, far, thresholds)
    eer = (frr_eer + far_eer) / 2

    tDCF_norm, cm_thresholds = compute_asvspoof_tDCF(
        asv_target_scores,
        asv_nontarget_scores,
        cm_target_scores,
        cm_nontarget_scores,
        cost_model
    )

    print("ASV EER and its threshold: {:.4f}\t{:.4f}".format(eer, eer_threshold))
    print("Normalized min-tDCF:       {:.4f}".format(np.min(tDCF_norm)))
