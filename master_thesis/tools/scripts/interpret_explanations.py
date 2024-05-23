import glob
import brainstat.datasets
import numpy as np
import matplotlib.pyplot as plt
import brainstat
import brainstat.datasets as datasets

from argparse import ArgumentParser
from brainstat.datasets import fetch_yeo_networks_metadata
from brainspace.utils.parcellation import map_to_labels
from brainstat.context.resting import yeo_networks_associations
from surfplot import Plot

from master_thesis.tools.plots.config import set_style, reset_style, COLUMN_HIGHT, COLUMN_WIDTH


LEFT_HEMISPHERE, RIGHT_HEMISPHERE = brainstat.datasets.fetch_template_surface('fsaverage5', join=False)


def plot_interpretation(explenations_array: np.ndarray, stroke: bool, n_stds: float = 3.0):
    suffix = "stroke" if stroke else "control"

    # Fetch the schaefer parcellation and the yeo networks
    schaefer = datasets.fetch_parcellation('fsaverage5','schaefer',100)
    network_names, yeo_colormap = fetch_yeo_networks_metadata(7)

    # Aggregate the explanations

    # Calculate the mean and std of the non-zero explanations
    explenations_summed = explenations_array.sum(axis=0)
    explanations_nonzero = explenations_summed[explenations_summed != 0]
    mean, std = explanations_nonzero.mean(), explanations_nonzero.std()

    # Count connection with importance greater than the threshold
    if stroke:
        thresholded = np.logical_and(explenations_summed > mean + n_stds * std, explenations_summed != 0)
    else:
        thresholded = np.logical_and(explenations_summed < mean - n_stds * std, explenations_summed != 0)
    important_edges = np.where(thresholded, 1, 0)
    print(f"({suffix}) Ratio of important edges: {important_edges.sum() / 100}%")

    # The degree of each ROI reflects the number of important connections, hence the importance of the ROI
    roi_degrees = important_edges.sum(axis=0)

    # Map ROI degrees to the schaefer parcellation
    schaefer_surf = map_to_labels(roi_degrees, schaefer, mask=schaefer!=0).astype(float)

    # Map ROI degrees to the yeo networks and compute mean degree for each network
    yeo_tstat_mean = yeo_networks_associations(schaefer_surf, "fsaverage5", seven_networks=True, reduction_operation="mean")

    # Plot the results 
    # - plot the brain surface with the sum aggregated explanations
    reset_style()
    plt.rcParams.update({
        'font.size': 9,
        'font.family': 'serif',
        'font.serif': ["Times"],
    })
    plot = Plot(LEFT_HEMISPHERE, RIGHT_HEMISPHERE)
    plot.add_layer(schaefer_surf, cbar=False)
    fig = plot.build()
    fig.savefig(f"xai_surfplot_{suffix}.pdf", bbox_inches="tight")
    plt.clf()

    # - plot the mean aggregated explanations for the yeo networks
    set_style()
    fig, ax = plt.subplots(1, 1, figsize=(COLUMN_WIDTH, COLUMN_HIGHT))
    ax.bar(
        np.arange(7),
        yeo_tstat_mean[:, 0],
        color=yeo_colormap,
        error_kw={"elinewidth": 5},
    )
    ax.set_xticks(np.arange(7), network_names, rotation=90)
    ax.set_ylabel("Mean degree")
    plt.savefig(f"xai_yeo_{suffix}.pdf", bbox_inches="tight")
    plt.clf()


if __name__ == "__main__":

    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument("-p", "--explentations_path", required=True, help="Path to the directory with explenations")
    parser.add_argument("-n", "--n_stds", default=2.0, help="Number of standard deviations to threshold the explenations")
    args = parser.parse_args()

    xs_stroke = np.array([np.load(f) for f in glob.glob(args.explentations_path+"/*PAT*")])
    xs_control = np.array([np.load(f) for f in glob.glob(args.explentations_path+"/*CON*")])

    plot_interpretation(xs_stroke, True, args.n_stds)
    plot_interpretation(xs_control, False, args.n_stds)

