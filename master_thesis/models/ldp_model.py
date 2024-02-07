from typing import Tuple, Callable

import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from torch_geometric.data import Data
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from master_thesis.utils.loading import ConnectivityDataset, get_balanced_train_test_masks
from master_thesis.utils.plotting import plot_confusion_matrix

# Constants
EC_DATASET_PATH = "/Users/wciezobka/sano/projects/masters-thesis/Datasets/NeuroFlicks/networks"
RESULTS_PATH = "/Users/wciezobka/sano/projects/masters-thesis/Results/ldp_model/"
CAUSAL_COEFF_THRESHOLD = 0.5
TEST_HOLDOUT = 32
K_FOLDS = 10
SEED = 42

# Hyperparameters
MAX_EPOCHS = int(1e4)
BATCH_SIZE = 'auto'
LEARNING_RATE = 1e-3
NUMBERS_OF_LAYERS = 2
HIDDEN_SIZE = 4
ALPHA = 2.0
OPTIMIZER = 'adam'
ACTIVATION = 'relu'
N_BINS = 10


def get_ldps(dataset: ConnectivityDataset) -> Tuple[np.ndarray, np.ndarray]:

    def aggregate_ldp(data: Data) -> np.ndarray:
        ldp = data.x
        x = [np.histogram(ldp[:, i], bins=N_BINS, density=True)[0] for i in range(5)]
        return np.concatenate(x, axis=0)
    
    xs = []
    ys = []
    for data in dataset:
        x = aggregate_ldp(data)
        xs.append(x)
        ys.append(data.y.item())
    xs = np.stack(xs, axis=0)
    ys = np.array(ys)

    return xs, ys


def log_experiment() -> str:

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join(RESULTS_PATH, timestamp)
    os.makedirs(results_dir, exist_ok=True)

    # Save hyperparameters
    hyperparameters = {
        "seed": SEED,
        "k_folds": K_FOLDS,
        "test_holdout": TEST_HOLDOUT,
        "max_epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "numbers_of_layers": NUMBERS_OF_LAYERS,
        "hidden_size": HIDDEN_SIZE,
        "alpha": ALPHA,
        "optimizer": OPTIMIZER,
        "activation": ACTIVATION,
        "n_bins": N_BINS,
    }
    with open(os.path.join(results_dir, "hyperparameters.json"), "w") as f:
        json.dump(hyperparameters, f)
    
    # Log to results.md
    with open(os.path.join(results_dir, "results.md"), "a") as f:

        # Log constants
        f.write(f"# Classification with Local Degree Profile\n")
        f.write(f"\n## Constants\n")
        f.write(f"EC dataset path: {EC_DATASET_PATH}\n")
        f.write(f"Causal coefficient threshold: {CAUSAL_COEFF_THRESHOLD}\n")
        f.write(f"Test holdout: {TEST_HOLDOUT}\n")
        f.write(f"K-folds: {K_FOLDS}\n")
        f.write(f"Seed: {SEED}\n")

        # Log hyperparameters
        f.write(f"\n## Hyperparameters\n")
        f.write(f"Max epochs: {MAX_EPOCHS}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n")
        f.write(f"Numbers of layers: {NUMBERS_OF_LAYERS}\n")
        f.write(f"Hidden size: {HIDDEN_SIZE}\n")
        f.write(f"Alpha: {ALPHA}\n")
        f.write(f"Optimizer: {OPTIMIZER}\n")
        f.write(f"Activation: {ACTIVATION}\n")
        f.write(f"Number of bins: {N_BINS}\n")
    
    return results_dir


def evaluate_model(
        ys_train_accumulator: np.ndarray,
        ys_hat_train_accumulator: np.ndarray,
        ys_test_accumulator: np.ndarray,
        ys_hat_test_accumulator: np.ndarray,
        results_dir_path: str
    ):
    
    def get_metric_stats(metric: Callable, ys_gold_matrix: np.ndarray, ys_hat_matrix: np.ndarray) -> Tuple[float, float]:
        metric_values = [metric(ys_gold, ys_hat) for ys_gold, ys_hat in zip(ys_gold_matrix, ys_hat_matrix)]
        return np.mean(metric_values), np.std(metric_values)
    
    cm_train_path = os.path.join(results_dir_path, "confusion_matrix_train.png")
    cm_test_path = os.path.join(results_dir_path, "confusion_matrix_test.png")
    
    # Evaluate the model on train set
    train_accuracy_mean, train_accuracy_std = get_metric_stats(accuracy_score, ys_train_accumulator, ys_hat_train_accumulator)
    train_precision_mean, train_precision_std = get_metric_stats(precision_score, ys_train_accumulator, ys_hat_train_accumulator)
    train_recall_mean, train_recall_std = get_metric_stats(recall_score, ys_train_accumulator, ys_hat_train_accumulator)
    train_f1_mean, train_f1_std = get_metric_stats(f1_score, ys_train_accumulator, ys_hat_train_accumulator)
    train_auc_mean, train_auc_std = get_metric_stats(roc_auc_score, ys_train_accumulator, ys_hat_train_accumulator)
    plot_confusion_matrix(ys_train_accumulator.flatten(), ys_hat_train_accumulator.flatten(), save_path=cm_train_path)
    
    # Evaluate the model on test set
    test_accuracy_mean, test_accuracy_std = get_metric_stats(accuracy_score, ys_test_accumulator, ys_hat_test_accumulator)
    test_precision_mean, test_precision_std = get_metric_stats(precision_score, ys_test_accumulator, ys_hat_test_accumulator)
    test_recall_mean, test_recall_std = get_metric_stats(recall_score, ys_test_accumulator, ys_hat_test_accumulator)
    test_f1_mean, test_f1_std = get_metric_stats(f1_score, ys_test_accumulator, ys_hat_test_accumulator)
    test_auc_mean, test_auc_std = get_metric_stats(roc_auc_score, ys_test_accumulator, ys_hat_test_accumulator)
    plot_confusion_matrix(ys_test_accumulator.flatten(), ys_hat_test_accumulator.flatten(), save_path=cm_test_path)

    # Save results
    with open(os.path.join(results_dir_path, "results.md"), "a") as f:
        f.write(f"\n## Train\n")
        f.write(f"Accuracy: {train_accuracy_mean:.3f} ± {train_accuracy_std:.3f}\n")
        f.write(f"Precision: {train_precision_mean:.3f} ± {train_precision_std:.3f}\n")
        f.write(f"Recall: {train_recall_mean:.3f} ± {train_recall_std:.3f}\n")
        f.write(f"F1: {train_f1_mean:.3f} ± {train_f1_std:.3f}\n")
        f.write(f"AUC: {train_auc_mean:.3f} ± {train_auc_std:.3f}\n")

        f.write(f"\n## Test\n")
        f.write(f"Accuracy: {test_accuracy_mean:.3f} ± {test_accuracy_std:.3f}\n")
        f.write(f"Precision: {test_precision_mean:.3f} ± {test_precision_std:.3f}\n")
        f.write(f"Recall: {test_recall_mean:.3f} ± {test_recall_std:.3f}\n")
        f.write(f"F1: {test_f1_mean:.3f} ± {test_f1_std:.3f}\n")
        f.write(f"AUC: {test_auc_mean:.3f} ± {test_auc_std:.3f}\n")


if __name__ == "__main__":

    # Log experiment
    results_dir_path = log_experiment()

    # Load networks
    networks = np.array(os.listdir(EC_DATASET_PATH))
    labels = np.array([np_file[4:7] == 'PAT' for np_file in networks])

    # K-fold cross validation
    ys_train_accumulator = []
    ys_hat_train_accumulator = []
    ys_test_accumulator = []
    ys_hat_test_accumulator = []
    for k in tqdm(range(K_FOLDS)):
        
        # Split data into train and test sets
        half_test_size = int(TEST_HOLDOUT/2)
        train_mask, test_mask = get_balanced_train_test_masks(labels, half_test_size=half_test_size, seed=(SEED+k))
        train_dataset = ConnectivityDataset(EC_DATASET_PATH, networks_mask=train_mask, causal_coeff_threshold=0.5, ldp=True)
        test_dataset = ConnectivityDataset(EC_DATASET_PATH, networks_mask=test_mask, causal_coeff_threshold=0.5, ldp=True)
        
        # Aggregate LDPs
        xs_train, ys_train = get_ldps(train_dataset)
        xs_test, ys_test = get_ldps(test_dataset)

        # Define MLP classifier
        clf_model = MLPClassifier(
            hidden_layer_sizes=(HIDDEN_SIZE,) * NUMBERS_OF_LAYERS,
            activation=ACTIVATION,
            solver=OPTIMIZER,
            alpha=ALPHA,
            batch_size=BATCH_SIZE,
            learning_rate="constant",
            learning_rate_init=LEARNING_RATE,
            max_iter=MAX_EPOCHS,
            random_state=SEED+k,
        )

        # Train MLP classifier
        clf_model.fit(xs_train, ys_train)

        # Predict on train and test sets
        ys_hat_train = clf_model.predict(xs_train)
        ys_hat_test = clf_model.predict(xs_test)

        # Accumulate results
        ys_train_accumulator.append(ys_train)
        ys_hat_train_accumulator.append(ys_hat_train)
        ys_test_accumulator.append(ys_test)
        ys_hat_test_accumulator.append(ys_hat_test)

    # Evaluate model
    evaluate_model(
        np.stack(ys_train_accumulator, axis=0),
        np.stack(ys_hat_train_accumulator, axis=0),
        np.stack(ys_test_accumulator, axis=0),
        np.stack(ys_hat_test_accumulator, axis=0),
        results_dir_path
    )
