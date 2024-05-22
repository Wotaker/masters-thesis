from argparse import ArgumentParser

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

from master_thesis.tools.plots import set_style

warnings.filterwarnings("ignore")


def test_hypothesis_II(results_dir: str, dataset_superior: str, dataset_inferior: str, pval_thr: float = 0.05, use_mapping: bool = True):
    """
    Hypothesis II: The AUC of the model trained on the RCC-based EC networks is significantly greater than
    the AUC of the model trained on the Granger-based EC networks. In other words, the whole-brain RCC method
    produces better EC networks than the Granger causality method.

    Parameters
    ----------
    results_dir : str
        Path to the results directory.
    dataset_superior : str
        Name of the superior dataset (RCC).
    dataset_inferior : str
        Name of the inferior dataset (GC or FC).
    pval_thr : float, optional
        The p-value threshold for statistical significance, by default 0.05.
    """

    mapping = {
        "stroke_RCC": "RCC",
        "stroke_GC": "GC",
    }
    
    # Load the results from the CSV file
    results_file = os.path.join(results_dir, "results.csv")
    results_df = pd.read_csv(results_file)
    results_df = results_df[
        (results_df.dataset.isin([dataset_superior, dataset_inferior])) &
        (results_df.split == "test") &
        (results_df.fold > 0)
    ]

    # Create an empty array to store the results
    models = sorted(results_df.model.unique(), reverse=True)
    comparison = np.zeros((len(models), len(models)))
    mask = np.logical_not(np.diag(np.ones(len(models), dtype=bool)))

    # Iterate over each model pair
    for i, model_superior in enumerate(models):
        for j, model_inferior in enumerate(models):

            # Filter the results dataframe for the specific models
            df_superior = results_df[(results_df.model == model_superior) & (results_df.dataset == dataset_superior)]
            df_inferior = results_df[(results_df.model == model_inferior) & (results_df.dataset == dataset_inferior)]

            # Perform two-sample t-test to verify hypothesis II
            # TODO Replace with https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
            _, pval = ttest_ind(df_superior.auc.values, df_inferior.auc.values, equal_var=False, alternative="greater")

            # Store the p-value in the comparison matrix
            comparison[i, j] = pval

            # Print the results
            print(f"{model_superior} vs. {model_inferior}: p-value = {pval}")
    
    # Plot the comparison matrix
    ax = sns.heatmap(
        comparison,
        vmin=0.,
        vmax=2*pval_thr,
        xticklabels=models,
        yticklabels=models,
        annot=True,
        fmt=".2E",
        square=True,
        mask=mask,
        cmap="coolwarm",
        center=pval_thr,
        # cbar=False,
        cbar_kws={"label": r"$p$-value"},
        annot_kws={"fontsize": 3}
    )
    ax.figure.subplots_adjust(left=0.3, bottom=0.5)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=30, ha="right")
    plt.xlabel(f"Inferior {mapping[dataset_inferior]}")
    plt.ylabel(f"Superior {mapping[dataset_superior]}")
    plt.savefig(os.path.join(results_dir, "hypothesis-II.pdf"), bbox_inches="tight")
    plt.clf()

if __name__ == "__main__":

    # Parse the command-line arguments
    parser = ArgumentParser()
    parser.add_argument("-r", "--results_dir", type=str, required=True, help="Path to the results directory.")
    parser.add_argument("-s", "--dataset_superior", type=str, required=True, help="Name of the superior dataset.")
    parser.add_argument("-i", "--dataset_inferior", type=str, required=True, help="Name of the inferior dataset.")
    parser.add_argument("-p", "--pval_thr", type=float, default=0.05, help="The p-value threshold for statistical significance.")
    args = parser.parse_args()

    # Test hypothesis II
    test_hypothesis_II(args.results_dir, args.dataset_superior, args.dataset_inferior, args.pval_thr)
