# -*- coding: utf-8 -*-

"""
Utilities for benchmarking performance and correctness of methods.

Author: Max A. Bowman
Date: 2/21/2024
"""

from typing import Tuple
import numpy as np
from libsvm.commonutil import svm_read_problem
from libsvm.svmutil import *


def sigma(x):
    return 1 / (1 + np.exp(-x))


def classification_accuracy(labels: np.array, X: np.array, theta: np.array) -> np.array:
    """

    :param labels: data labels in {-1, 1}.
    :param X: data matrix with features as columns.
    :param theta:
    :return:
    """
    labels_hat = sigma(X @ theta)
    # print(labels_hat)
    labels_hat = (labels_hat >= 0.5).astype(np.float32)
    labels_hat = 2*labels_hat - 1  # convert to {-1, 1}
    # print(labels)
    # print(labels_hat)

    diff = labels_hat - labels
    num_different = np.count_nonzero(diff)

    return (len(labels) - num_different) / len(labels)


def libsvm_to_numpy(filename: str, num_features: int) -> Tuple[np.array, np.array]:
    labels, features = svm_read_problem(filename)
    data_matrix = np.zeros((len(features), num_features + 1))

    for i in range(len(features)):
        for j in features[i].keys():
            data_matrix[i, j - 1] = features[i][j]

    data_matrix[:, -1] = 1  # for bias

    labels = np.array(labels)
    labels[labels == 0] = -1.0
    labels[labels == 2] = -1.0

    labels = labels.reshape((len(labels),))

    data_matrix = data_matrix.astype(np.float32)
    labels = labels.astype(np.float32)

    return data_matrix, labels
