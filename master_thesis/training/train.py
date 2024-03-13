from typing import List, Dict, Tuple, Optional
from argparse import ArgumentParser

import datetime
import os
import yaml
import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.model_selection import KFold
from karateclub.graph_embedding import Graph2Vec
from tqdm import tqdm

from master_thesis.utils import LOGLEVEL_MAP
from master_thesis.classification_models import BaseModel, MODELS_MAP
from master_thesis.tools.data import  load_np_data, Preprocessing
from master_thesis.tools.plots import plot_sample_networks


def load_dataset(dataset_name: str, dataset_config: Dict) -> Tuple[List[nx.DiGraph], List[int]]:

    # Extract dataset configuration
    networks_dir_path = dataset_config["path"]
    channel = dataset_config.pop("channel", None)
    preprocesing_kwargs = dataset_config["preprocessing"]

    # Load and preprocess networks
    X, y = Preprocessing(seed=global_seed, **preprocesing_kwargs)(*load_np_data(networks_dir_path, channel))
    logging.info(f"Loaded {len(X)} networks with {len(X[0].nodes)} nodes each")

    # Plot sample networks
    if logging.getLogger().level <= LOGLEVEL_MAP["PLOT"]:
        logging.log(LOGLEVEL_MAP["PLOT"], "Plotting sample networks...")
        plot_sample_networks(X, y, rows=4, save_path=f"{dataset_name}_sample_networks.png")
    
    # Return dataset
    return X, y


def holdout(model_name: str, model_config: Dict, X: List[nx.DiGraph], y: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    # Extract model configuration
    holdout_size = int(model_config["holdout_size"] * len(X))
    hyperparameters = model_config.pop("hyperparameters", {})

    # Shuffle data
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = [X[i] for i in idx]
    y = [y[i] for i in idx]

    # Split data
    X_train, X_test = X[:-holdout_size], X[-holdout_size:]
    y_gold_train, y_gold_test = y[:-holdout_size], y[-holdout_size:]

    # Train model
    model: BaseModel = MODELS_MAP[model_name](**hyperparameters)
    model.fit(X_train, y_gold_train)

    # Predict
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)

    # Return
    return y_gold_train, y_hat_train, y_gold_test, y_hat_test


def kfold(model_name: str, model_config: Dict, X: List[nx.DiGraph], y: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    # Extract model configuration
    folds = model_config["folds"]
    hyperparameters = model_config.pop("hyperparameters", {})

    # Define accumulator lists
    y_gold_train_acc, y_hat_train_acc = [], []
    y_gold_test_acc, y_hat_test_acc = [], []

    # Define k-fold cross-validation
    kfold_cv = KFold(n_splits=folds, shuffle=True, random_state=global_seed)
    for fold, (train_index, test_index) in tqdm(enumerate(kfold_cv.split(X)), total=folds, desc="Cross-validation"):

        # Split data
        X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
        y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

        # Train model (LTP)
        model: BaseModel = MODELS_MAP[model_name](**hyperparameters)
        model.fit(X_train, y_train)

        # Predict
        y_hat_train = model.predict(X_train)
        y_hat_train_acc.append(y_hat_train)
        y_gold_train_acc.append(y_train)

        y_hat_test = model.predict(X_test)
        y_hat_test_acc.append(y_hat_test)
        y_gold_test_acc.append(y_test)
    
    # Concatenate lists
    y_hat_train = np.concatenate(y_hat_train_acc)
    y_gold_train = np.concatenate(y_gold_train_acc)

    y_hat_test = np.concatenate(y_hat_test_acc)
    y_gold_test = np.concatenate(y_gold_test_acc)

    # Return
    return y_gold_train, y_hat_train, y_gold_test, y_hat_test


def create_experiment_dir(dir_path: str, config_path: str) -> None:

    # Get timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create directory
    dir_path = os.path.join(dir_path, timestamp)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # Copy configuration file to experiment directory
    os.system(f"cp {config_path} {os.path.join(dir_path, 'config.yaml')}")
    
    # Return
    return dir_path


def log_results(
        results: pd.DataFrame,
        root_dir: str,
        dataset_name: str,
        model_name: str,
        y_gold_train: np.ndarray,
        y_hat_train: np.ndarray,
        y_gold_test: np.ndarray,
        y_hat_test: np.ndarray
    ):

    # Create model results directory
    model_results_dir_path = os.path.join(root_dir, f"{dataset_name}_{model_name}")
    os.makedirs(model_results_dir_path, exist_ok=True)

    # Calculate train and test metrices
    train_metrices = BaseModel.evaluate(
        y_gold_train,
        y_hat_train,
        save_path=os.path.join(model_results_dir_path, "cm_train.png")
    )
    test_metrices = BaseModel.evaluate(
        y_gold_test,
        y_hat_test,
        save_path=os.path.join(model_results_dir_path, "cm_test.png")
    )

    # Log results
    results = results.append({
        "dataset": dataset_name,
        "model": model_name,
        "split": "train",
        "accuracy": train_metrices.accuracy,
        "precision": train_metrices.precision,
        "recall": train_metrices.recall,
        "f1": train_metrices.f1,
        "auc": train_metrices.auc
    }, ignore_index=True)
    results = results.append({
        "dataset": dataset_name,
        "model": model_name,
        "split": "test",
        "accuracy": test_metrices.accuracy,
        "precision": test_metrices.precision,
        "recall": test_metrices.recall,
        "f1": test_metrices.f1,
        "auc": test_metrices.auc
    }, ignore_index=True)

    return results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True)
    parser.add_argument("-r", "--results_dir", type=str, default="Results")
    parser.add_argument("-ll", "--log_level", type=str, default="INFO")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=LOGLEVEL_MAP[args.log_level.upper()])

    # Load configuration
    with open(args.config_path, 'r') as stream:
        config = yaml.load(stream.read(), Loader=yaml.FullLoader)
    
    # Set global seed
    global_seed = config["seed"]
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)

    # Set up results directory and results dataframe
    experiment_dir_path = create_experiment_dir(args.results_dir, args.config_path)
    logging.info(f"Created experiment directory at `{experiment_dir_path}`")
    results = pd.DataFrame(columns=["dataset", "model", "split", "accuracy", "precision", "recall", "f1", "auc" ])

    # Loop over datasets
    for dataset_name, dataset_config in config["datasets"].items():
        logging.info(f"Loading dataset {dataset_name}...")

        # Load dataset
        X, y = load_dataset(dataset_name, dataset_config)

        # Loop over models
        for model_name, model_config in dataset_config["models"].items():
            logging.info(f"Training model {model_name} on dataset {dataset_name}...")
            
            # When using Graph2Vec, embed graphs first
            if model_name.lower() == "graph2vec":
                logging.info("Embedding graphs with Graph2Vec...")

                graph2vec_kwargs = model_config.pop("hyperparameters", {})
                graph2vec = Graph2Vec(**graph2vec_kwargs)
                graph2vec.fit(X)
                X = graph2vec.get_embedding()

            # If holdout size is specified, use holdout validation
            if model_config.get("holdout_size", None) is not None:
                logging.info("Using holdout validation...")
                y_gold_train, y_hat_train, y_gold_test, y_hat_test = holdout(model_name.upper(), model_config, X, y)
            
            # If folds are specified, use k-fold cross-validation
            elif model_config.get("folds", None) is not None:
                logging.info(f"Using {model_config['folds']}-fold cross-validation...")
                y_gold_train, y_hat_train, y_gold_test, y_hat_test = kfold(model_name.upper(), model_config, X, y)
            
            # If neither holdout size nor folds are specified, log error and skip
            else:
                logging.error(f"Invalid configuration for model {model_name} in dataset {dataset_name}." +\
                              "Specify either 'holdout_size' or 'folds'. Skipping...")
                continue

            # Gather model results
            results = log_results(
                results,
                experiment_dir_path,
                dataset_name,
                model_name,
                y_gold_train,
                y_hat_train,
                y_gold_test,
                y_hat_test
            )
    
    # Save results
    results.to_csv(os.path.join(experiment_dir_path, "results.csv"), index=False)
            

    