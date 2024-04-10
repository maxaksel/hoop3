# -*- coding: utf-8 -*-

"""
Synthetic logistic regression comparison entry point.

Author: Max A. Bowman
Date: 12/31/2023
"""

import numpy as np
import matplotlib.pyplot as plt
from funcs.logistic import sigma, LogisticFunctions
from opt.opt1 import gradient_descent, acc_gradient_descent
from opt.opt2 import newton_plain, cubic_newton_slow, cubic_newton


def main():
    d = 250  # dimension of feature space
    N = 1000  # number of data points

    # Data matrix (features are columns) randomly generated with uniform distribution [-1, 1]
    X = 2 * (np.random.rand(N, d + 1) - 0.5)
    X[:, -1] = 1
    X = X.astype(np.float32)

    # Theta vector
    weights = np.random.rand(d + 1)
    weights /= np.linalg.norm(weights)  # normalize weights
    weights = weights.astype(np.float32)

    # Labels vector
    y = 2 * (sigma(X @ weights.T) > 0.5) - 1  # generate labels in {-1, 1}
    y = y.astype(np.float32)

    # Define logistic regression problem
    logistic_problem = LogisticFunctions(data_matrix=X, labels=y, use_gpu=False)

    start_theta = np.random.rand(d + 1)
    start_theta /= np.linalg.norm(start_theta)
    start_theta = start_theta.astype(np.float32)

    # Run optimizers
    print("Performing gradient descent...")
    loss_o1, theta_star = gradient_descent(start_theta, 1000, logistic_problem)
    print("Performing accelerated gradient descent...")
    loss_o1_a, theta_star = acc_gradient_descent(start_theta, 1000, logistic_problem)

    print("Performing Newton's method with quadratic regularization...")
    loss_o2_q, theta_star = newton_plain(start_theta, 1000, logistic_problem, 10, 1)
    print("Performing Newton's method with cubic regularization...")
    loss_o2_c, theta_star = cubic_newton_slow(start_theta, 1000, logistic_problem)
    print("Performing accelerated Newton's method with cubic regularization...")
    loss_o2_ac, theta_star = cubic_newton(start_theta, 1000, logistic_problem)

    # Plot loss over iterations
    plt.semilogy(loss_o1, label="Gradient Descent")
    plt.semilogy(loss_o1_a, label="Acc. Gradient Descent")
    plt.semilogy(loss_o2_q, label="Quad. Newton")
    plt.semilogy(loss_o2_c, label="Cubic Newton")
    plt.semilogy(loss_o2_ac, label="Acc. Cubic Newton")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
