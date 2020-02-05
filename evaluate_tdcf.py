# Evaluate joint t-DCF of asv and cm systems
import argparse
import os

import numpy as np

from utils import tdcf
from utils.score_loading import load_scorefile_and_split_scores, load_scorefile_and_split_to_arrays
from utils.plot_utils import plot_score_distribution, plot_joint_score_scatter

parser = argparse.ArgumentParser("Compute t-DCFs on ASV and CM scorelists")
parser.add_argument("asv_scores", help="ASV scorelist")
parser.add_argument("cm_scores", help="CM scorelist")
parser.add_argument("--plots", type=str, help="Location where to store different plots")
parser.add_argument("--experiment-name", type=str, default="", help="Name to include in files etc")


def main_tdcf(args):
    """
    Read scores from ASV and CM scorelists, compute tDCFs and
    show results
    """
    # Score-lists are assumed to be lines with format
    #   is_target score [optional ...]
    asv_is_target, asv_scores, asv_systems = load_scorefile_and_split_to_arrays(args.asv_scores)
    asv_original_scores = asv_scores.astype(np.float32)
    asv_is_target = asv_is_target == "True"
    asv_scores = asv_scores.astype(np.float32)

    # Take spoof scores as well
    asv_spoof_scores = asv_scores[asv_systems != "bonafide"]
    # Only use proper ASV samples (non-spoof) to determine
    # EER for ASV
    asv_is_target_bonafide = asv_is_target[asv_systems == "bonafide"]
    asv_scores = asv_scores[asv_systems == "bonafide"]
    # Split to target/non-target scores
    asv_scores = asv_scores.astype(np.float32)
    asv_target_scores = asv_scores[asv_is_target_bonafide]
    asv_nontarget_scores = asv_scores[~asv_is_target_bonafide]

    # Get the cm target listing and target/nontarget scores
    cm_is_target = load_scorefile_and_split_to_arrays(args.cm_scores)[0]
    cm_is_target = cm_is_target == "True"
    cm_target_scores, cm_nontarget_scores, cm_original_scores = load_scorefile_and_split_scores(args.cm_scores)

    if args.plots is not None:
        plot_score_distribution(
            asv_target_scores,
            asv_nontarget_scores,
            args.experiment_name + "_asv",
            filename=os.path.join(args.plots, args.experiment_name + "_scores_asv.png")
        )

        plot_score_distribution(
            cm_target_scores,
            cm_nontarget_scores,
            args.experiment_name + "_cm",
            filename=os.path.join(args.plots, args.experiment_name + "_scores_cm.png")
        )

        plot_joint_score_scatter(
            asv_original_scores,
            cm_original_scores,
            asv_is_target,
            cm_is_target,
            args.experiment_name + "_joint",
            filename=os.path.join(args.plots, args.experiment_name + "_scores_joint.png")
        )

    # ASVSpoof19's way of computing t-DCF: 
    #  Fix ASV thresholds to EER, vary CM thresholds and
    #  pick minimum t-DCF.
    #  Note that only bonafide samples are used to determine
    #  ASV EER

    # Compute the EER and EER threshold for ASV we are going to use   
    asv_frr, asv_far, asv_thresholds = tdcf.compute_det(asv_target_scores, asv_nontarget_scores)
    asv_frr_eer, asv_far_eer, asv_eer_threshold = tdcf.compute_eer(asv_frr, asv_far, asv_thresholds)
    asv_eer = (asv_frr_eer + asv_far_eer) / 2

    # Compute CM EER too as we want to plot it
    cm_frr, cm_far, cm_thresholds = tdcf.compute_det(cm_target_scores, cm_nontarget_scores)
    cm_frr_eer, cm_far_eer, cm_eer_threshold = tdcf.compute_eer(cm_frr, cm_far, cm_thresholds)
    cm_eer = (cm_frr_eer + cm_far_eer) / 2

    tDCF_norm, cm_thresholds = tdcf.compute_asvspoof_tDCF(
        asv_target_scores,
        asv_nontarget_scores,
        asv_spoof_scores,
        cm_target_scores,
        cm_nontarget_scores,
        tdcf.ASVSPOOF2019_COST_MODEL
    )

    print("ASV EER and its threshold:   {:.4f}\t{:.4f}".format(asv_eer, asv_eer_threshold))
    print("CM EER:                      {:.4f}".format(cm_eer))
    print("Normalized min-tDCF:         {:.4f}".format(np.min(tDCF_norm)))


if __name__ == "__main__":
    args = parser.parse_args()
    main_tdcf(args)
