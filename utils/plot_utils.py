import warnings
from itertools import product
import os

import numpy as np
import scipy
from tqdm import tqdm
from matplotlib import pyplot
from matplotlib.animation import FFMpegWriter
from sklearn.manifold import TSNE
import umap

# Prior fo the sidekit rocch_det plot function
PRIOR = 0.001
HIST_BINS = 30

def compute_eer(
    target_scores,
    non_target_scores
):
    """
    Compute Equal Error Rate using SIDEKIT's functions
    """
    # Import SIDEKIT here to avoid it messing up other stuff
    import sidekit
    eer = sidekit.bosaris.fast_minDCF(target_scores, non_target_scores, 0)[-1]
    return eer


def plot_score_distribution(
    target_scores,
    non_target_scores,
    experiment_name,
    hist_bins=HIST_BINS,
    filename=None,
):
    """
    Plot score distribution of target scores and non-target scores.
    If filename is given, save the plot to this location.

    Returns: Figure object which includes the plot.
    """

    fig, ax = pyplot.subplots()
    ax2 = ax.twinx()

    ax2.hist(target_scores, bins=hist_bins, density=True, color="blue",
                alpha=0.5)
    ax.hist(non_target_scores, bins=hist_bins, density=True, color="red",
                alpha=0.5)

    ax.set_xlabel("Score", )
    ax2.set_ylabel("Target normed count", color="b")
    ax.set_ylabel("Non-target normed count", color="r")
    ax.set_title(experiment_name)
    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename)

    return fig


def plot_joint_score_scatter(
    asv_scores,
    cm_scores,
    asv_is_target,
    cm_is_target,
    experiment_name,
    filename=None,
):
    """
    Plot joint ASV and CM scores in a scatterplot, colored by
    three different classes:
        0 = Target samples (bonafide)
        1 = Nontarget bonafide
        2 = Spoof samples
    cm_is_target/asv_is_target tell these labels.
    If filename is given, save the plot to this location.

    Returns: Figure object which includes the plot.
    """

    fig, ax = pyplot.subplots(figsize=[6.4*3, 4.8*3], dpi=300)

    # First plot negative samples because there tends to be
    # more of these and they would just cover positive samples
    # First nontarget speakers (but bonafide)
    nontarget_idxs = (~asv_is_target) & cm_is_target
    ax.scatter(asv_scores[nontarget_idxs], cm_scores[nontarget_idxs], c="r", s=10,
               alpha=0.5, edgecolors="none", linewidth=0)
    # Spoof samples
    spoof_idxs = ~cm_is_target
    ax.scatter(asv_scores[spoof_idxs], cm_scores[spoof_idxs], c="g", s=10,
               alpha=0.5, edgecolors="none", linewidth=0)
    # Target samples
    target_idxs = asv_is_target & cm_is_target
    ax.scatter(asv_scores[target_idxs], cm_scores[target_idxs], c="b", s=10,
               alpha=0.5, edgecolors="none", linewidth=0)

    ax.set_xlabel("ASV score")
    ax.set_ylabel("CM score")
    ax.legend(("Nontarget", "Spoof", "Target"))
    ax.set_title(experiment_name)
    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename)

    return fig


def plot_joint_score_scatter_animation(
    asv_scores_list,
    cm_scores_list,
    asv_is_target_list,
    cm_is_target_list,
    titles_list,
    filename,
    fps=5,
):
    """
    Similar to plot_joint_score_scatter, but for writing
    animations.

    asv_scores, cm_scores, asv_is_target, cm_is_target and
    titles are expected to be lists of equal length, each
    item representing one frame.

    Based off on this Matplotlib example:
        https://matplotlib.org/3.1.1/gallery/animation/frame_grabbing_sgskip.html
    """
    pyplot.rcParams['animation.ffmpeg_path'] = "ffmpeg"

    fig, ax = pyplot.subplots(figsize=[6.4*3, 4.8*3], dpi=200)

    writer = FFMpegWriter(fps=fps, bitrate=10000)
    num_frames = len(asv_scores_list)

    # Fix x-lim and y-lim for clarity
    max_asv = max([max(x) for x in asv_scores_list])
    min_asv = min([min(x) for x in asv_scores_list])
    max_cm = max([max(x) for x in cm_scores_list])
    min_cm = min([min(x) for x in cm_scores_list])

    with writer.saving(fig, filename, dpi=200):
        # Loop over frames and repeat drawing on all of them
        for frame_idx in tqdm(range(num_frames), desc="render"):
            # Clear the current plot

            ax.clear()
            # Pick right data
            asv_scores = asv_scores_list[frame_idx]
            cm_scores = cm_scores_list[frame_idx]
            asv_is_target = asv_is_target_list[frame_idx]
            cm_is_target = cm_is_target_list[frame_idx]
            title = titles_list[frame_idx]

            nontarget_idxs = (~asv_is_target) & cm_is_target
            ax.scatter(asv_scores[nontarget_idxs], cm_scores[nontarget_idxs], c="r", s=10,
                       alpha=0.5, edgecolors="none", linewidth=0)
            # Spoof samples
            spoof_idxs = ~cm_is_target
            ax.scatter(asv_scores[spoof_idxs], cm_scores[spoof_idxs], c="g", s=10,
                       alpha=0.5, edgecolors="none", linewidth=0)
            # Target samples
            target_idxs = asv_is_target & cm_is_target
            ax.scatter(asv_scores[target_idxs], cm_scores[target_idxs], c="b", s=10,
                       alpha=0.5, edgecolors="none", linewidth=0)

            ax.set_xlabel("ASV score")
            ax.set_ylabel("CM score")
            ax.set_xlim((min_asv, max_asv))
            ax.set_ylim((min_cm, max_cm))
            ax.legend(("Nontarget", "Spoof", "Target"))
            ax.set_title(title)

            writer.grab_frame()


def plot_joint_score_scatter_animation_snapshots(
    asv_scores_list,
    cm_scores_list,
    asv_is_target_list,
    cm_is_target_list,
    titles_list,
    filename,
    num_snapshots=5,
):
    """
    Similar to plot_joint_score_scatter_animation, but
    instead of writing a full video this will take
    snapshots and save them as PDFs. Designed for creating
    plots for the paper.

    asv_scores, cm_scores, asv_is_target, cm_is_target and
    titles are expected to be lists of equal length, each
    item representing one frame.

    Based off on this Matplotlib example:
        https://matplotlib.org/3.1.1/gallery/animation/frame_grabbing_sgskip.html
    """
    fig, ax = pyplot.subplots(figsize=[6.4, 6.4])

    # Fix x-lim and y-lim for clarity
    max_asv = max([max(x) for x in asv_scores_list])
    min_asv = min([min(x) for x in asv_scores_list])
    max_cm = max([max(x) for x in cm_scores_list])
    min_cm = min([min(x) for x in cm_scores_list])

    # Select points from which we create plots
    num_scores = len(asv_scores_list)

    plot_points = [int(i * (num_scores - 1) / (num_snapshots - 1)) for i in range(num_snapshots)]

    for i, frame_idx in enumerate(plot_points):
        # Clear the current plot
        ax.clear()
        # Pick right data
        asv_scores = asv_scores_list[frame_idx]
        cm_scores = cm_scores_list[frame_idx]
        asv_is_target = asv_is_target_list[frame_idx]
        cm_is_target = cm_is_target_list[frame_idx]
        title = titles_list[frame_idx]

        # Spoof samples
        spoof_idxs = ~cm_is_target
        ax.scatter(asv_scores[spoof_idxs], cm_scores[spoof_idxs], c="g", s=15,
                   alpha=1.0, edgecolors="none", linewidth=0)
        # Non-targets
        nontarget_idxs = (~asv_is_target) & cm_is_target
        ax.scatter(asv_scores[nontarget_idxs], cm_scores[nontarget_idxs], c="r", s=15, 
                   alpha=1.0, edgecolors="none", linewidth=0)
        # Target samples
        target_idxs = asv_is_target & cm_is_target
        ax.scatter(asv_scores[target_idxs], cm_scores[target_idxs], c="b", s=15,
                   alpha=1.0, edgecolors="none", linewidth=0)

        # No labels for for paper
        ax.set_xlim((min_asv, max_asv))
        ax.set_ylim((min_cm, max_cm))
        ax.tick_params(axis='both', which='both', labelsize=27)

        # Plot legend only to first plot
        if i == 0:
            # Trick stolen from Stackoverflow #24706125
            # to increase size of ticks in legend
            lgnd = ax.legend(("Spoof", "Nontarget", "Target"), prop={"size": 29})
            lgnd.legendHandles[0]._sizes = [50]
            lgnd.legendHandles[1]._sizes = [50]
            lgnd.legendHandles[2]._sizes = [50]

        fig.tight_layout()
        fig.savefig(filename.replace(".", "_%d." % frame_idx))


def plot_joint_score_training_progress_single_axis(
    num_updates,
    asv_eers,
    cm_eers,
    min_tdcfs,
    filename,
    asv_eers_stds=None,
    cm_eers_stds=None,
    min_tdcfs_stds=None,
    for_paper=False,
):
    """
    Plot a metrics curve over number of updates, including
    ASV and CM EERs as well as the minimum t-DCF. This
    version plots everything on single y-axis, i.e. change in
    ASV and CM is harder to see.

    Note: This one assumes the values are relative changes, because
          otherwise it does not make sense to plot them into one
          plot like this

    Additional parameters:
        *_stds: Draw shaded area around the main curve with plus/minus
                this std value
    If *_stds are provided, draw shaded area around the main curve with
    plus/minus this std value.

    If relative is true, change axis labels accordingly.
    """
    fig = None
    ax = None
    fig, ax = pyplot.subplots(figsize=[6.4, 6.4])

    asv_curve, = ax.plot(num_updates, asv_eers, color="blue")
    if asv_eers_stds is not None:
        ax.fill_between(
            num_updates,
            asv_eers - asv_eers_stds,
            asv_eers + asv_eers_stds,
            color="blue",
            alpha=0.2,
            linewidth=0
        )

    cm_curve, = ax.plot(num_updates, cm_eers, color="green")
    if cm_eers_stds is not None:
        ax.fill_between(
            num_updates,
            cm_eers - cm_eers_stds,
            cm_eers + cm_eers_stds,
            color="green",
            alpha=0.2,
            linewidth=0
        )

    tdcf_curve, = ax.plot(num_updates, min_tdcfs, color="red")
    if min_tdcfs_stds is not None:
        ax.fill_between(
            num_updates,
            min_tdcfs - min_tdcfs_stds,
            min_tdcfs + min_tdcfs_stds,
            color="red",
            alpha=0.2,
            linewidth=0,
        )

    ax.set_ylim((-75, 25))

    if for_paper and "ce" in filename:
        # Adjust axis for "CE experiments", because they are odd
        ax.set_ylim((-25, 75))

    ax.grid(True)
    ax.grid(alpha=0.25)
    # Plot line at zero
    ax.axhline(0, alpha=1.0, c="k")

    ax.tick_params(axis='both', which='both', labelsize=24)
    # Show plot info only in simple-eval plot (bottom left corner)
    if for_paper and "simple" in filename and "eval" in filename:
        ax.set_xlabel("Number of updates", fontsize=28)

        ax.legend(
            [asv_curve, cm_curve, tdcf_curve],
            ["ASV EER", "CM EER", "min t-DCF'"],
            prop={'size': 24}
        )
    else:
        # Empty padding, otherwise plot size changes
        ax.set_xlabel(" ", fontsize=28)
        ax.set_ylabel(" \n ", fontsize=28)

    # If left-most plot, include y-axis legend
    if for_paper and "simple" in filename:
        if "dev" in filename:
            ax.set_ylabel("Development set\nRelative change (%)", fontsize=28)
        if "eval" in filename:
            ax.set_ylabel("Evaluation set\nRelative change (%)", fontsize=28)

    # Add titles to topmost figures (development set)
    if for_paper and "dev" in filename:
        if "pg_simple" in filename:
            ax.set_title("REINFORCE\nSimple", fontsize=30)
        elif "pg_penalize" in filename:
            ax.set_title("REINFORCE\nPenalize", fontsize=30)
        elif "pg_reward" in filename:
            ax.set_title("REINFORCE\nReward", fontsize=30)
        elif "pg_tdcf" in filename:
            ax.set_title("REINFORCE\nt-DCF", fontsize=30)
        elif "ce_epochs" in filename:
            ax.set_title("Independent models\nSame labels", fontsize=30)
        elif "ce_split" in filename:
            ax.set_title("Independent models\nSeparate labels", fontsize=30)
    else:
        pass
        ax.set_title(" \n ")

    fig.tight_layout()
    fig.savefig(filename, bbox_inches='tight', transparent=True)


def plot_joint_score_training_progress(
    num_updates,
    asv_eers,
    cm_eers,
    min_tdcfs,
    filename,
    asv_eers_stds=None,
    cm_eers_stds=None,
    min_tdcfs_stds=None,
    relative=False,
):
    """
    Plot a metrics curve over number of updates, including
    ASV and CM EERs as well as the minimum t-DCF. Designed
    to be used with score animation plotting tool for better
    image on how joint training affected the two systems.
    
    Additional parameters:
        *_stds: Draw shaded area around the main curve with plus/minus
                this std value
        relative: Change axis labels to reflect relative change
        for_paper: Do modifications for the figures for paper plots
                   (e.g. larger font)
    If *_stds are provided, draw shaded area around the main curve with
    plus/minus this std value.

    If relative is true, change axis labels accordingly.
    """
    # Copypasta/tips from
    # https://matplotlib.org/3.1.1/gallery/ticks_and_spines/multiple_yaxis_with_spines.html
    fig, ax = pyplot.subplots()
    fig.subplots_adjust(right=0.75)

    asv_ax = ax.twinx()
    cm_ax = ax.twinx()

    # Offset the right spine of cm_ax.  The ticks and label have already been
    # placed on the right by twinx above.
    cm_ax.spines["right"].set_position(("axes", 1.2))
    # Having been created by twinx, cm_ax has its frame off, so the line of its
    # detached spine is invisible.  First, activate the frame but make the patch
    # and spines invisible.
    cm_ax.set_frame_on(True)
    cm_ax.patch.set_visible(False)
    for sp in cm_ax.spines.values():
        sp.set_visible(False)
    # Second, show the right spine.
    cm_ax.spines["right"].set_visible(True)

    asv_ax.plot(num_updates, asv_eers, color="blue")
    if asv_eers_stds is not None:
        asv_ax.fill_between(
            num_updates,
            asv_eers - asv_eers_stds,
            asv_eers + asv_eers_stds,
            color="blue",
            alpha=0.2,
            linewidth=0
        )

    cm_ax.plot(num_updates, cm_eers, color="green")
    if cm_eers_stds is not None:
        cm_ax.fill_between(
            num_updates,
            cm_eers - cm_eers_stds,
            cm_eers + cm_eers_stds,
            color="green",
            alpha=0.2,
            linewidth=0
        )

    ax.plot(num_updates, min_tdcfs, color="red")
    if min_tdcfs_stds is not None:
        ax.fill_between(
            num_updates,
            min_tdcfs - min_tdcfs_stds,
            min_tdcfs + min_tdcfs_stds,
            color="red",
            alpha=0.2,
            linewidth=0,
        )

    ax.set_xlabel("Number of updates", )
    if relative:
        ax.set_ylabel("Min t-DCF", color="red")
        asv_ax.set_ylabel("ASV EER", color="blue")
        cm_ax.set_ylabel("CM EER", color="green")
    else:
        ax.set_ylabel("Min t-DCF' (relative)", color="red")
        asv_ax.set_ylabel("ASV EER (relative)", color="blue")
        cm_ax.set_ylabel("CM EER (relative)", color="green")

    # Only include title if not for a paper
    ax.set_title(os.path.basename(filename), fontsize="small")

    fig.tight_layout()

    fig.savefig(filename)


def plot_det(
    target_scores,
    non_target_scores,
    system_names,
    experiment_name,
    filename=None,
    prior=PRIOR
):
    """
    target_scores, non_target_scores and system_names are Lists, one
    item for one curve.

    Plots DET curve using SIDEKIT, including Doddington’s Rule of 30 points
    for both FA and misses.
    If filename is given, save the figure in this location.

    Returns: Figure object which includes the plot.
    """
    # Import SIDEKIT here to avoid it messing anything else up
    import sidekit
    # Straight from the example:
    #  http://www-lium.univ-lemans.fr/sidekit/tutorial/rsr2015_gmm_ubm.html
    dp = sidekit.bosaris.DetPlot(
        window_style="sre10",
        plot_title=experiment_name
    )
    dp.create_figure()
    for i in range(len(target_scores)):
        dp.set_system(target_scores[i], non_target_scores[i], system_names[i])
        dp.plot_rocch_det(0, target_prior=prior)
        dp.plot_DR30_both(idx=0)
    dp.__figure__.tight_layout()

    if filename is not None:
        dp.__figure__.savefig(filename)

    return dp.__figure__


def plot_joint_det_animation(
    asv_scores_list,
    cm_scores_list,
    asv_is_target_list,
    cm_is_target_list,
    titles_list,
    filename,
    fps=5,
):
    """
    Plotting animations of the DET curves separately.

    asv_scores, cm_scores, asv_is_target, cm_is_target and
    titles are expected to be lists of equal length, each
    item representing one frame.

    Based off on this Matplotlib example:
        https://matplotlib.org/3.1.1/gallery/animation/frame_grabbing_sgskip.html
    """
    from utils import tdcf
    pyplot.rcParams['animation.ffmpeg_path'] = "ffmpeg"

    fig, ax = pyplot.subplots()

    writer = FFMpegWriter(fps=fps, bitrate=10000)
    num_frames = len(asv_scores_list)

    with writer.saving(fig, filename, dpi=200):
        # Loop over frames and repeat drawing on all of them
        for frame_idx in tqdm(range(num_frames), desc="render"):
            # Clear the current plot

            ax.clear()
            # Pick right data
            asv_scores = asv_scores_list[frame_idx]
            cm_scores = cm_scores_list[frame_idx]
            asv_is_target = asv_is_target_list[frame_idx]
            cm_is_target = cm_is_target_list[frame_idx]
            title = titles_list[frame_idx]

            # Compute DET curves
            # Test ASV against legit samples.

            ax.set_yscale("log")
            ax.set_xscale("log")

            asv_frr, asv_far, asv_thresholds = tdcf.compute_det(
                asv_scores[asv_is_target & cm_is_target],
                asv_scores[(~asv_is_target) & cm_is_target]
            )

            cm_frr, cm_far, cm_thresholds = tdcf.compute_det(
                cm_scores[cm_is_target],
                cm_scores[~cm_is_target]
            )

            # Turn ratios into percentages
            ax.plot(asv_far * 100, asv_frr * 100, c="b")
            ax.plot(cm_far * 100, cm_frr * 100, c="g")

            ax.set_xlabel("False Acceptance Rate (%)")
            ax.set_ylabel("False Rejection Rate (%)")
            ax.set_xlim((0.01, 100))
            ax.set_ylim((0.01, 100))
            ax.legend(("ASV", "CM"))
            ax.set_title(title, fontsize="small")

            writer.grab_frame()


def plot_joint_det_change(
    asv_scores_list,
    cm_scores_list,
    asv_is_target_list,
    cm_is_target_list,
    titles_list,
    filename,
):
    """
    Plot comparison of DET curves, from the beginning to the end

    asv_scores, cm_scores, asv_is_target, cm_is_target and
    titles are expected to be lists of equal length, each
    item representing one frame.

    Based off on this Matplotlib example:
        https://matplotlib.org/3.1.1/gallery/animation/frame_grabbing_sgskip.html
    """
    from utils import tdcf

    fig, (ax_asv, ax_cm) = pyplot.subplots(2, 1, "all", "all", figsize=[3.2, 6.4])

    ax_asv.set_yscale("log")
    ax_asv.set_xscale("log")

    ax_asv.set_xlim((0.1, 100))
    ax_asv.set_ylim((0.1, 100))

    # Plot initial positions
    asv_scores = asv_scores_list[0]
    cm_scores = cm_scores_list[0]
    asv_is_target = asv_is_target_list[0]
    cm_is_target = cm_is_target_list[0]
    asv_frr, asv_far, asv_thresholds = tdcf.compute_det(
        asv_scores[asv_is_target & cm_is_target],
        asv_scores[(~asv_is_target) & cm_is_target]
    )

    cm_frr, cm_far, cm_thresholds = tdcf.compute_det(
        cm_scores[cm_is_target],
        cm_scores[~cm_is_target]
    )

    # Turn ratios into percentages
    ax_asv.plot(asv_far * 100, asv_frr * 100, ":", c="b", linewidth=2)
    ax_cm.plot(cm_far * 100, cm_frr * 100, ":", c="g", linewidth=2)

    # Plot after training
    asv_scores = asv_scores_list[-1]
    cm_scores = cm_scores_list[-1]
    asv_is_target = asv_is_target_list[-1]
    cm_is_target = cm_is_target_list[-1]
    asv_frr, asv_far, asv_thresholds = tdcf.compute_det(
        asv_scores[asv_is_target & cm_is_target],
        asv_scores[(~asv_is_target) & cm_is_target]
    )

    cm_frr, cm_far, cm_thresholds = tdcf.compute_det(
        cm_scores[cm_is_target],
        cm_scores[~cm_is_target]
    )

    # Turn ratios into percentages
    ax_asv.plot(asv_far * 100, asv_frr * 100, c="b", linewidth=2)
    ax_cm.plot(cm_far * 100, cm_frr * 100, c="g", linewidth=2)

    # Hardcoded trickery for paper
    if "simple" in filename:
        ax_asv.set_ylabel("False Rejection Rate (%)", fontsize=14)
        ax_cm.set_ylabel("False Rejection Rate (%)", fontsize=14)
        ax_cm.set_xlabel("False Acceptance Rate (%)", fontsize=14)
    else:
        ax_asv.set_ylabel(" ", fontsize=14)
        ax_cm.set_ylabel(" ", fontsize=14)
        ax_cm.set_xlabel(" ", fontsize=14)


    if "simple" in filename:
        ax_asv.legend(("Before", "After"), prop={"size": 15})
        ax_cm.legend(("Before", "After"), prop={"size": 15})

    ax_asv.set_title("ASV", fontsize=18)
    ax_cm.set_title("CM", fontsize=18)

    ax_asv.tick_params(axis='both', which='both', labelsize=14)
    ax_cm.tick_params(axis='both', which='both', labelsize=14)

    fig.tight_layout()
    fig.savefig(filename, transparent=True)


def create_marker_color_cycle(markers=("o", "X", "P")):
    """
    Create an iterator that will return tuples (color, marker)
    to be used for plotting many different classes with
    scatter
    """
    colors = pyplot.rcParams["axes.prop_cycle"].by_key()["color"]

    # Vary color first, then markers
    marker_color_iterator = product(markers, colors)

    return marker_color_iterator


def plot_tsne(
    arrays,
    legends,
    filename=None,
    use_umap=False,
    **kwargs
):
    """
    Plot tSNE and return the figure.

    Parameters:
        arrays (List of ndarray): Vectors to be plotted. Each
            array will be colored and named differently.
        legends (List of str): Name for each array in the plot
        filename (str): Where to save the figure (default: None)
        use_umap (bool): Use UMAP instead of tSNE
        **kwargs: Parameters fed to tSNE/UMAP
    Returns:
        figure (Figure): Matplotlib Figure with the tSNE/umap plot.
        arrays (List of ndarray): Transformer vectors in same order as argument
                                  arrays
    """
    # Concetenate arrays into one and
    # create a label array
    vectors = np.concatenate(arrays, axis=0)
    label_array = np.concatenate([np.zeros((len(arrays[i],))) + i for i in range(len(arrays))])

    # Apply tSNE/UMAP
    manifold_learner_class = TSNE
    if use_umap:
        manifold_learner_class = umap.UMAP

    manifold_learner = manifold_learner_class(n_components=2, **kwargs)

    # This block filters out warnings. UMAP (numba under the hood) prints out
    # quite a bit of warnings we might not want to see.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vectors = manifold_learner.fit_transform(vectors)

    fig, ax = pyplot.subplots(figsize=[6.4*3, 4.8*3], dpi=300)

    # Use longer iterator for colors and markers
    marker_color_iterator = create_marker_color_cycle()

    # Plot results
    # Using separate scatter calls to avoid the mess with
    # coloring one specific plot
    for i in range(len(arrays)):
        marker, color = next(marker_color_iterator)
        plot_vectors = vectors[label_array == i]
        ax.scatter(
            plot_vectors[:, 0],
            plot_vectors[:, 1],
            s=10,
            alpha=0.5,
            edgecolors="none",
            linewidth=0,
            c=color,
            marker=marker
        )

    ax.legend(legends)

    if filename is not None:
        fig.savefig(filename)

    return_arrays = [vectors[label_array == i] for i in range(len(arrays))]

    return fig, return_arrays
