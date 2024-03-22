from argparse import ArgumentParser

import os
import warnings
import pandas as pd
from scipy.stats import ttest_1samp

warnings.filterwarnings("ignore")


def test_hypothesis_I(results_dir: str):
    """
    Hypothesis I: The AUC of the model is significantly greater than 0.5. In other words, the model is better than random.

    Parameters
    ----------
    results_dir : str
        Path to the results directory.
    """

    # Load the results from the CSV file
    results_file = os.path.join(results_dir, "results.csv")
    results_df = pd.read_csv(results_file)

    # Create an empty dataframe from dict to store the results
    results_summary_dict = {
        "dataset": [],
        "model": [],
        "combined_auc": [],
        "mean_auc": [],
        "std_auc": [],
        "pval_auc": [],
        "mean_accuracy": [],
        "std_accuracy": [],
        "mean_precision": [],
        "std_precision": [],
        "mean_recall": [],
        "std_recall": [],
        "mean_f1": [],
        "std_f1": [],
    }

    # Iterate over each dataset and model
    for dataset in results_df.dataset.unique():
        for model in results_df.model.unique():

            # Filter the results dataframe for the specific dataset and model
            df = results_df[(results_df.dataset == dataset) & (results_df.model == model) & (results_df.split == "test")]

            # Extract the metrics
            combined_auc = df[df.fold == 0].auc
            df = df[df.fold > 0]
            accuracy = df.accuracy.values
            precision = df.precision.values
            recall = df.recall.values
            f1 = df.f1.values
            auc = df.auc.values

            # Perform one-sample t-test to verify hypothesis I
            _, pval = ttest_1samp(auc, 0.5, alternative="greater")

            # Store the results in the summary dictionary
            results_summary_dict["pval_auc"].append(pval)
            results_summary_dict["combined_auc"].append(combined_auc.values[0])
            results_summary_dict["dataset"].append(dataset)
            results_summary_dict["model"].append(model)
            results_summary_dict["mean_accuracy"].append(accuracy.mean())
            results_summary_dict["std_accuracy"].append(accuracy.std())
            results_summary_dict["mean_precision"].append(precision.mean())
            results_summary_dict["std_precision"].append(precision.std())
            results_summary_dict["mean_recall"].append(recall.mean())
            results_summary_dict["std_recall"].append(recall.std())
            results_summary_dict["mean_f1"].append(f1.mean())
            results_summary_dict["std_f1"].append(f1.std())
            results_summary_dict["mean_auc"].append(auc.mean())
            results_summary_dict["std_auc"].append(auc.std())

    # Convert the summary dictionary to a dataframe
    results_summary_df = pd.DataFrame(results_summary_dict)
    
    # Save the results to a CSV file
    results_summary_df.to_csv(os.path.join(results_dir, "results_summary.csv"), index=False)

if __name__ == "__main__":

    # Parse the command-line arguments
    parser = ArgumentParser()
    parser.add_argument("-r", "--results_dir", type=str, required=True, help="Path to the results directory.")
    args = parser.parse_args()

    # Test hypothesis I
    test_hypothesis_I(args.results_dir)