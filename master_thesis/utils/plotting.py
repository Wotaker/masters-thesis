import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import *


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

def evaluate(y_train, y_train_hat, y_test, y_test_hat):

    print(f"Train accuracy: {accuracy_score(y_train, y_train_hat):.2f},\t\tTest accuracy: {accuracy_score(y_train, y_train_hat):.2f}")
    print(f"Train recall: {recall_score(y_train, y_train_hat):.2f}, Test recall: {recall_score(y_test, y_test_hat):.2f}")
    print(f"Train precision: {precision_score(y_train, y_train_hat):.2f}, Test precision: {precision_score(y_test, y_test_hat):.2f}")
    print(f"Train f1: {f1_score(y_train, y_train_hat):.2f}, Test f1: {f1_score(y_test, y_test_hat):.2f}")
    print(f"Train AUC: {roc_auc_score(y_train, y_train_hat):.2f}, Test AUC: {roc_auc_score(y_test, y_test_hat):.2f}")
    plot_confusion_matrix(y_test, y_test_hat)