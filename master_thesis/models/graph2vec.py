import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

from karateclub.graph_embedding import Graph2Vec

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split #, GridSearchCV, LeaveOneOut, learning_curve
from sklearn.metrics import *

from master_thesis.utils.loading import load_networks
from master_thesis.utils.plotting import *

DATASET_PATH = 'Datasets/SynapseSnap/ischemic-subjects/'
SEED = 44       # Good seds: 44, 50, 61
CAUSAL_COEFF_STRENGTH = 1.5
HIDDEN_DIM = 16
TEST_SIZE = 0.3
ALPHA = 100.0


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
    # rc = RidgeClassifier(alpha=100.)
    rc = MLPClassifier(hidden_layer_sizes=(32, 16, 8), max_iter=100, random_state=SEED)
    rc.fit(X_train, y_train)
    y_train_hat, y_test_hat = rc.predict(X_train), rc.predict(X_test)

    # Evaluate the model
    evaluate(y_train, y_train_hat, y_test, y_test_hat)

    
