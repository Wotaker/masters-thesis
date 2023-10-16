import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

from karateclub.graph_embedding import Graph2Vec

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
# from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split #, GridSearchCV, LeaveOneOut, learning_curve
from sklearn.metrics import *

from master_thesis.utils.load_networks import load_networks

DATASET_PATH = 'Datasets/SynapseSnap/ischemic-cortical-subjects/'
SEED = 44       # Good seds: 44, 50, 61
CAUSAL_COEFF_STRENGTH = 1.5
HIDDEN_DIM = 16
TEST_SIZE = 0.3
ALPHA = 100.0

# Utility plotting functions
def plot_confusion_matrix(y_true, y_pred, save_path=None):

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix")
    plt.savefig(save_path) if save_path else plt.show()

def plot_2d_embeddings(embeddings, labels, save_path=None):
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(embeddings[labels == 0, 0], embeddings[labels == 0, 1], c='tab:blue', s=5, alpha=0.5, label='control')
    ax.scatter(embeddings[labels == 1, 0], embeddings[labels == 1, 1], c='tab:orange', s=5, alpha=0.5, label='patological')
    ax.set_title('Graph2Vec embeddings')
    plt.legend()
    plt.savefig(save_path) if save_path else plt.show()

if __name__ == '__main__':

    # Load networks as networkx graphs
    dataset, labels, (n_control, n_patological) = load_networks(
        DATASET_PATH,
        causal_coeff_strength=CAUSAL_COEFF_STRENGTH,
        verbose=True
    )

    # Define embedding model embed networks onto vectors
    model = Graph2Vec(dimensions=HIDDEN_DIM, wl_iterations=2, epochs=200, seed=SEED, workers=1)
    model.fit(dataset)
    embeddings = model.get_embedding()

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=TEST_SIZE, random_state=42)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"Ratio of positive samples in train: {y_train.mean()}")
    print(f"Ratio of positive samples in test: {y_test.mean()}")

    # Train Ridge classifier on the embeddings
    rc = RidgeClassifier(alpha=100.)
    rc.fit(X_train, y_train)
    y_train_hat, y_test_hat = rc.predict(X_train), rc.predict(X_test)

    # Calculate metrices
    # train_accuracy = (y_train_hat == y_train).mean()
    # test_accuracy = (y_test_hat == y_test).mean()
    print(f"Train accuracy: {accuracy_score(y_train, y_train_hat):.2f},\t\tTest accuracy: {accuracy_score(y_train, y_train_hat):.2f}")
    print(f"Train recall: {recall_score(y_train, y_train_hat):.2f},\t\tTest recall: {recall_score(y_test, y_test_hat):.2f}")
    print(f"Train precision: {precision_score(y_train, y_train_hat):.2f},\t\tTest precision: {precision_score(y_test, y_test_hat):.2f}")
    print(f"Train f1: {f1_score(y_train, y_train_hat):.2f},\t\t\tTest f1: {f1_score(y_test, y_test_hat):.2f}")
    print(f"Train AUC: {roc_auc_score(y_train, y_train_hat):.2f},\t\tTest AUC: {roc_auc_score(y_test, y_test_hat):.2f}")
    plot_confusion_matrix(y_test, y_test_hat, save_path='confusion_matrix.png')

    
