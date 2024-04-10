from functools import partial
from typing import Dict, List, Callable, Optional

import os
import logging
import yaml
import numpy as np
import pandas as pd
import torch
from lime import lime_tabular
from tqdm import tqdm
from argparse import ArgumentParser

from master_thesis.utils import LOGLEVEL_MAP
from master_thesis.classification_models import BaseModel, MODELS_MAP
from master_thesis.training.train import create_experiment_dir, load_dataset, log_results


class ExplainWithLIME:

    def __init__(
            self,
            X_train: np.ndarray,
            X_test: np.ndarray,
            subject_names_test: List[str],
            predict_fn: Callable,
            experiment_dir_path: str,
            top_n_features: int = 10,
            seed: int = 42,
            nodes: int = 100
    ) -> None:

        # Set up the class attributes
        self.nodes = nodes
        self.n_features = nodes*nodes
        self.X_train_normalized_flatten = X_train.reshape(-1, self.n_features)
        self.X_test_normalized_flatten = X_test.reshape(-1, self.n_features)
        self.subject_names_test = subject_names_test
        self.predict_fn = predict_fn
        self.experiment_dir_path = experiment_dir_path
        self.top_n_features = top_n_features
        self.seed = seed

        # create a LIME explainer
        self.explainer = lime_tabular.LimeTabularExplainer(
            self.X_train_normalized_flatten,
            mode="classification",
            feature_names=list(range(self.n_features)),
            class_names=["Control", "Pathological"],
            discretize_continuous=False,
            random_state=self.seed
        )

        # Prepare directory for explanations
        os.makedirs(os.path.join(self.experiment_dir_path, "explanations"), exist_ok=True)
    
    def explain_subject(
            self,
            x: np.ndarray,
            sub: str,
            top_n_features: Optional[int] = None,
            save: bool = True
    ) -> np.ndarray:

        # Prepare the empty heatmap for the subject explanation
        subject_explain_heatmap = np.zeros((self.n_features,))

        # Explain the subject
        exp = self.explainer.explain_instance(
            x,
            self.predict_fn,
            labels=(1,),
            num_features=top_n_features if top_n_features is not None else self.top_n_features
        )

        # Extract the LIME scores and fill the heatmap
        lime_scores = exp.local_exp[1]
        for i, score in lime_scores:
            subject_explain_heatmap[i] = score
        subject_explain_heatmap = subject_explain_heatmap.reshape(self.nodes, self.nodes)

        # Save and return the explanation heatmap
        if save:
            np.save(os.path.join(self.experiment_dir_path, "explanations", f"{sub}.npy"), subject_explain_heatmap)
        return subject_explain_heatmap
    
    def explain_testset(
        self,
        top_n_features: Optional[int] = None,
        save: bool = True
    ) -> np.ndarray:
        
        # Explain each subject and accumulate the heatmaps
        explain_heatmap_sum = np.zeros((self.nodes, self.nodes))
        for x, sub in tqdm(zip(self.X_test_normalized_flatten, self.subject_names_test), total=len(self.X_test_normalized_flatten)):
            explain_heatmap = self.explain_subject(x, sub, top_n_features, save)
            explain_heatmap_sum += explain_heatmap

        # Save and return the accumulated heatmap
        if save:
            np.save(os.path.join(self.experiment_dir_path, "explanations", "sum-heatmap.npy"), explain_heatmap_sum)
        return explain_heatmap_sum


def train_model(config: Dict, experiment_dir_path: str, results_df: pd.DataFrame, global_seed: int):

    # Load dataset
    dataset_name, dataset_config = list(config["datasets"].items())[0]
    logging.info(f"Loading dataset {dataset_name}...")
    dataset_config["preprocessing"]["seed"] = global_seed
    X, y, filenames = load_dataset(dataset_name, dataset_config, with_filenames=True)
    subject_names = [filename.split(".")[0] for filename in filenames]

    # Model definition
    model_name, model_config = list(dataset_config["models"].items())[0]
    logging.info(f"Training model {model_name} on dataset {dataset_name}...")

    # - If holdout size is specified, use holdout validation
    if model_config.get("holdout_size", None) is None:
        logging.error("Please use holdout validation for LIME analysis!")
    
    # Extract model configuration
    holdout_size = int(model_config["holdout_size"] * len(X))
    hyperparameters = model_config.pop("hyperparameters", {})

    # Shuffle data
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = np.array([X[i] for i in idx])
    y = np.array([y[i] for i in idx])
    subject_names = [subject_names[i] for i in idx]

    # Split data
    X_train, X_test = X[:-holdout_size], X[-holdout_size:]
    y_gold_train, y_gold_test = y[:-holdout_size], y[-holdout_size:]
    subject_names_test = subject_names[-holdout_size:]

    # Subtraction of control mean
    if dataset_config["subtract_mean"]:
        total_control_mean = X[y == 0].mean(axis=0)
        train_control_mean = X_train[y_gold_train == 0].mean(axis=0)
        X_test_normalized = X_test - total_control_mean
        X_train_normalized = X_train - train_control_mean

    # Train model
    model: BaseModel = MODELS_MAP[model_name](**hyperparameters)
    model.fit(X_train, y_gold_train, dataset_config)

    # Predict
    y_hat_train = model.predict(
        X_train_normalized,
        dataset_config
    )
    y_hat_test = model.predict(X_test_normalized, dataset_config)

    logging.info(f"Model {model_name} has finished training on dataset {dataset_name}, saving results...")

    # Log final results
    results_df = log_results(
        results_df,
        experiment_dir_path,
        dataset_name,
        model_name,
        0,
        y_gold_train,
        y_hat_train,
        y_gold_test,
        y_hat_test
    )

    # Save results
    results_df.to_csv(os.path.join(experiment_dir_path, "results.csv"), index=False)

    # Define the predict_proba function
    predict_proba_fn = partial(model.predict_proba, dataset_config=dataset_config)

    return X_train_normalized, X_test_normalized, subject_names_test, predict_proba_fn


if __name__ == "__main__":

    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument("-c", "--config_path", required=True, help="Path to the configuration file")
    parser.add_argument("-r", "--results_dir", default=os.path.join("Results", "lime"))
    parser.add_argument("-ll", "--log_level", type=str, default="INFO")
    args = parser.parse_args()

    config_path = args.config_path
    results_dir = args.results_dir

    # Set up logging
    logging.basicConfig(level=LOGLEVEL_MAP[args.log_level.upper()])

    # Load experiment configuration
    with open(config_path, 'r') as stream:
        config = yaml.load(stream.read(), Loader=yaml.FullLoader)

    # Set global seed
    global_seed = config["seed"]
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)

    # Set up results directory and results dataframe
    experiment_dir_path = create_experiment_dir(results_dir, config_path)
    logging.info(f"Created experiment directory at `{experiment_dir_path}`")
    results_df = pd.DataFrame(columns=["dataset", "model", "fold", "split", "accuracy", "precision", "recall", "f1", "auc" ])

    # Train model
    X_train_normalized, X_test_normalized, subject_names_test, predict_proba_fn = train_model(
        config, experiment_dir_path, results_df, global_seed
    )

    # Explain with LIME
    explain_with_lime = ExplainWithLIME(
        X_train_normalized,
        X_test_normalized[:3],
        subject_names_test,
        predict_proba_fn,
        experiment_dir_path,
        top_n_features=100,
        seed=global_seed
    )
    logging.info("Explaining test set with LIME...")
    explain_with_lime.explain_testset()




