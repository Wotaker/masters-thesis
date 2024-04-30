from master_thesis.tools.data import  load_np_data, Preprocessing
from master_thesis.tools.plots import *
import argparse


def figure_methods_3(networks_dir_path: str):

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
        connection_weight_threshold=(0.6, -0.35)
    )(X_norm, y)

    # Generate figures
    plot_aggregated_weight_histogram(X_raw, bins=50, xtitle="Edge weight", save_path="methods_3a.pdf")
    plot_aggregated_weight_histogram(X_normalized, bins=50, xtitle="Edge weight", save_path="methods_3b.pdf")
    plot_aggregated_weight_histogram(X_processed, bins=50, xtitle="Edge weight", save_path="methods_3c.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--networks_dir_path", type=str, required=True, help="Path to the networks directory")
    args = parser.parse_args()

    figure_methods_3(args.networks_dir_path)
