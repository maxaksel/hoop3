# -*- coding: utf-8 -*-

"""
Synthetic logistic regression comparison entry point.

Author: Max A. Bowman
Date: 1/17/2024
"""

import numpy as np
from opt.opt1 import gradient_descent, acc_gradient_descent, acc_gradient_descent2
from opt.opt2 import newton_plain, cubic_newton_slow, cubic_newton, acc_cubic_newton, acc_cubic_newton2, newton_plain_no_param
from opt.opt3 import hyperfast
import matplotlib.pyplot as plt
from funcs.logistic import sigma, LogisticFunctions
import time


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


def l3_compute(y: np.array, X: np.array, theta: np.array) -> float:
    """
    Used for exploration of Lipschitz constant of third order.
    :param y: labels.
    :param X: data matrix.
    :param theta: weights.
    :return: lower bound of L3.
    """
    N = len(y)  # number of data points
    F = X.shape[1]  # number of features

    z = y*(X@theta)

    lower_bound = 0.0

    for w in range(F):
        for j in range(F):
            for k in range(F):
                for l in range(F):
                    sum_wjkl = 0
                    for i in range(N):
                        sum_wjkl += X[i, w]*X[i, j]*X[i, k]*X[i, l]*((np.exp(-z[i])-3*np.exp(-3*z[i]))*(1+np.exp(-z[i]))-4*np.exp(-z[i])*(np.exp(-z[i])-np.exp(-3*z[i])))/np.power(1 + np.exp(z[i]), 5)
                        lower_bound += sum_wjkl**2

    return lower_bound/N


if __name__ == '__main__':
    d = 25  # dimension of feature space
    N = 100  # number of data points
    X = (2 * (np.random.rand(N, d + 1) - 0.5)).astype(np.float32)  # data matrix (features are columns) randomly generated with uniform distribution [-1, 1]
    X[:, -1] = 1
    weights = np.random.rand(d + 1)
    weights /= np.linalg.norm(weights)  # normalize weights

    y = 2 * (sigma(X @ weights.T) > 0.5) - 1  # generate labels in {-1, 1}
    y = y.astype(np.float32)

    classification_accuracy(y, X, weights)

    logistic_problem = LogisticFunctions(data_matrix=X, labels=y, use_gpu=True)

    num_iters = 50

    # Gradient Descent
    print("Gradient Descent\n=================")
    theta_hat = np.random.rand(d + 1).astype(np.float32)
    orig_theta_hat = np.copy(theta_hat)

    start_time = time.time()
    loss_o1_q, _, theta_hat = gradient_descent(theta_hat, num_iters, logistic_problem)
    end_time = time.time()
    print("Run time:")
    print(end_time - start_time)
    print(loss_o1_q[-1])

    # plt.semilogy(loss_o1_q.cpu())
    # plt.xlabel(r"Iterations")
    # plt.ylabel(r"$l(\theta; X, y)$")
    # plt.show()

    print("Classification accuracy:")
    print(classification_accuracy(y, X, theta_hat))

    # Accelerated Gradient Descent
    print("Acc. Gradient Descent\n=================")
    theta_hat = np.copy(orig_theta_hat)
    start_time = time.time()
    loss_o1_aq, theta_hat = acc_gradient_descent2(theta_hat, num_iters, logistic_problem)
    end_time = time.time()
    print("Run time:")
    print(end_time - start_time)
    print(loss_o1_aq[-1])

    # plt.semilogy(loss_o1_aq.cpu())
    # plt.xlabel(r"Iterations")
    # plt.ylabel(r"$l(\theta; X, y)$")
    # plt.show()

    print("Classification accuracy:")
    print(classification_accuracy(y, X, theta_hat))

    # Newton
    # theta_hat = np.copy(orig_theta_hat)
    # start_time = time.time()
    # loss_o2_newton, theta_hat = newton_plain_no_param(theta_hat, 200, logistic_problem)
    # end_time = time.time()
    # print("Newton\n=================")
    # print("Run time:")
    # print(end_time - start_time)
    # print(loss_o2_newton[-1])

    # plt.semilogy(loss_o2_q.cpu())
    # plt.xlabel(r"Iterations")
    # plt.ylabel(r"$l(\theta; X, y)$")
    # plt.show()

    # Quadratic Newton
    print("Quadratic Newton\n=================")
    theta_hat = np.copy(orig_theta_hat)
    start_time = time.time()
    loss_o2_q, _, theta_hat = newton_plain(theta_hat, num_iters, logistic_problem, alpha=10, gamma=1)
    end_time = time.time()
    print("Run time:")
    print(end_time - start_time)
    print(loss_o2_q[-1])

    # plt.semilogy(loss_o2_q.cpu())
    # plt.xlabel(r"Iterations")
    # plt.ylabel(r"$l(\theta; X, y)$")
    # plt.show()

    print("Classification accuracy:")
    print(classification_accuracy(y, X, theta_hat))

    # Cubic Newton
    print("Cubic Newton Slow\n=================")
    theta_hat = np.copy(orig_theta_hat)
    start_time = time.time()
    loss_o2_c, _, theta_hat = cubic_newton_slow(theta_hat, num_iters, logistic_problem)
    end_time = time.time()
    print("Run time:")
    print(end_time - start_time)
    print(loss_o2_c[-1])

    print("Classification accuracy:")
    print(classification_accuracy(y, X, theta_hat))

    # Fast Cubic Newton
    print("Fast Cubic Newton\n=================")
    theta_hat = np.copy(orig_theta_hat)
    start_time = time.time()
    loss_o2_cf, _, theta_hat = cubic_newton(theta_hat, num_iters, logistic_problem)
    end_time = time.time()
    print("Run time:")
    print(end_time - start_time)

    print(loss_o2_cf[-1])

    # plt.semilogy(loss_o1_q.cpu())
    # plt.xlabel(r"Iterations")
    # plt.ylabel(r"$l(\theta; X, y)$")
    # plt.show()

    print("Classification accuracy:")
    print(classification_accuracy(y, X, theta_hat))

    # Accelerated Cubic Newton
    theta_hat = np.copy(orig_theta_hat)
    start_time = time.time()
    loss_o2_accel, grad_o2_accel, theta_hat = acc_cubic_newton(theta_hat, num_iters, logistic_problem)
    end_time = time.time()
    print("Accelerated Cubic Newton\n=================")
    print("Run time:")
    print(end_time - start_time)

    print(loss_o2_accel[-1])

    # plt.semilogy(loss_o1_q.cpu())
    # plt.xlabel(r"Iterations")
    # plt.ylabel(r"$l(\theta; X, y)$")
    # plt.show()

    print("Classification accuracy:")
    print(classification_accuracy(y, X, theta_hat))

    # Third-Order Slow Hyperfast
    theta_hat = np.copy(orig_theta_hat)
    start_time = time.time()
    loss_o3_accel, grad_o3_accel, theta_hat = hyperfast(theta_hat, num_iters, logistic_problem)
    end_time = time.time()
    print("Hyperfast Slow\n=================")
    print("Run time:")
    print(end_time - start_time)

    print(loss_o3_accel[-1])

    # plt.semilogy(loss_o1_q.cpu())
    # plt.xlabel(r"Iterations")
    # plt.ylabel(r"$l(\theta; X, y)$")
    # plt.show()

    print("Classification accuracy:")
    print(classification_accuracy(y, X, theta_hat))

    # Combined Graph
    plt.semilogy(loss_o1_q, label="Gradient Descent")
    plt.semilogy(loss_o1_aq, label="Acc. Gradient Descent")
    # plt.semilogy(loss_o2_newton, label="Newton")
    plt.semilogy(loss_o2_q, label="Quadratic Newton")
    plt.semilogy(loss_o2_c, label="Cubic Newton Slow")
    plt.semilogy(loss_o2_cf, label="Cubic Newton Fast")
    # plt.semilogy(loss_o2_accel, label="Acc. Cubic Newton")
    plt.semilogy(loss_o3_accel, label="Hyperfast Slow")
    plt.xlabel(r"Iterations")
    plt.ylabel(r"$l(\theta; X, y)$")
    plt.legend()
    plt.show()
