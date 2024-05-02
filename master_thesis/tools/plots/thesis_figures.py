import os
import pandas as pd

from master_thesis.tools.data import  load_np_data, Preprocessing
from master_thesis.tools.plots import *


TIMESERIES_BEFORE_DIR = "/Users/wciezobka/sano/projects/Effective-Connectivity-Reservoir-Computing/Datasets/stroke"
TIMESERIES_AFTER_DIR = "/Users/wciezobka/sano/projects/Effective-Connectivity-Reservoir-Computing/Datasets/stroke-split"
RCC_NETWORKS_DIR = "/Users/wciezobka/sano/projects/masters-thesis/Datasets/NeuroFlicksUnidirRCC/networks"
GC_NETWORKS_DIR = "/Users/wciezobka/sano/projects/masters-thesis/Datasets/NeuroFlicksGC/networks"


def figure_methods_2(timeseries_dir_path_before: str, timeseries_dir_path_after: str, histogram_bins: int = 20):

    def dataset_histogram(dir_path: str, legend_location: str, save_path: str):

        # List all subject files in the directory
        subject_files = os.listdir(dir_path)
        subject_files = [f for f in subject_files if f.endswith('.tsv')]

        # Define helper functions
        timesteps = lambda file: len(pd.read_csv(os.path.join(dir_path, file), sep='\t'))
        subject_name = lambda file: file.split('_')[0]
        subject_type = lambda file: "PAT" in file.split('_')[0]

        # Create dataframe with subject lengths
        lengths_df = list(map(lambda file: (file, subject_name(file), timesteps(file), subject_type(file)), subject_files))
        lengths_df = pd.DataFrame(lengths_df, columns=['FileName', 'SubjectName', 'Length', 'Patological'])
        lengths_df = lengths_df.sort_values(by='Length', ascending=False)

        # Renema column Patological to "Subject Type"
        lengths_df_hist = lengths_df.rename(columns={"Patological": "Subject type"})

        # Count total number of timeseries
        total_timeseries = lengths_df_hist.shape[0]
        total_pat = lengths_df_hist[lengths_df_hist["Subject type"] == 1].shape[0]
        total_con = lengths_df_hist[lengths_df_hist["Subject type"] == 0].shape[0]

        # Assign labels to the subject type
        label_con = f"Control\n({total_con} in total)"
        label_pat = f"Pathological\n({total_pat} in total)"

        lengths_df_hist["Subject type"] = lengths_df_hist["Subject type"].apply(lambda x: label_pat if x else label_con)

        sns.histplot(
            data=lengths_df_hist,
            x='Length',
            hue='Subject type',
            hue_order=[label_con, label_pat],
            bins=histogram_bins,
            multiple="dodge",
            shrink=0.8
        )
        sns.move_legend(plt.gca(), legend_location)
        plt.title(f"{total_timeseries} time series in total")
        plt.savefig(save_path, bbox_inches='tight')
        plt.clf()
    
    dataset_histogram(timeseries_dir_path_before, legend_location="upper left", save_path="methods_2a.pdf")
    dataset_histogram(timeseries_dir_path_after, legend_location="upper right", save_path="methods_2b.pdf")


def figure_methods_3(networks_dir_path: str, histogram_bins: int = 50):

    # Load raw numpy networks
    X, y = load_np_data(networks_dir_path, channel=0)
    X, y = np.array(X), np.array(y)

    # Subtract control mean
    control_mean = X[y == 0].mean(axis=0)
    X_norm = X - control_mean

    # Preprocess data
    X_raw, _ = Preprocessing(
        undirected=False,
    )(X, y)
    X_normalized, _ = Preprocessing(
        undirected=False,
    )(X_norm, y)
    X_processed, _ = Preprocessing(
        undirected=False,
        connection_weight_threshold=(0.25, -0.55)
    )(X_norm, y)

    # Generate figures
    plot_aggregated_weight_histogram(X_raw, bins=histogram_bins, xtitle="Edge weight", save_path="methods_3a.pdf")
    plot_aggregated_weight_histogram(X_normalized, bins=histogram_bins, xtitle="Edge weight", save_path="methods_3b.pdf")
    plot_aggregated_weight_histogram(X_processed, bins=histogram_bins, xtitle="Edge weight", save_path="methods_3c.pdf")


if __name__ == "__main__":

    figure_methods_2(TIMESERIES_BEFORE_DIR, TIMESERIES_AFTER_DIR)
    figure_methods_3(RCC_NETWORKS_DIR)
