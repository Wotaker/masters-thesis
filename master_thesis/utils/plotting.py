from sklearn.metrics import *

from tools.plots import plot_confusion_matrix


def evaluate(y_train, y_train_hat, y_test, y_test_hat):

    print(f"Train accuracy: {accuracy_score(y_train, y_train_hat):.2f},\t\tTest accuracy: {accuracy_score(y_train, y_train_hat):.2f}")
    print(f"Train recall: {recall_score(y_train, y_train_hat):.2f}, Test recall: {recall_score(y_test, y_test_hat):.2f}")
    print(f"Train precision: {precision_score(y_train, y_train_hat):.2f}, Test precision: {precision_score(y_test, y_test_hat):.2f}")
    print(f"Train f1: {f1_score(y_train, y_train_hat):.2f}, Test f1: {f1_score(y_test, y_test_hat):.2f}")
    print(f"Train AUC: {roc_auc_score(y_train, y_train_hat):.2f}, Test AUC: {roc_auc_score(y_test, y_test_hat):.2f}")
    plot_confusion_matrix(y_test, y_test_hat)