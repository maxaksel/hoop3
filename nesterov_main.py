# -*- coding: utf-8 -*-

"""
Entry point for benchmarking with Nesterov's difficult functions for tensor methods.

Author: Max A. Bowman
Date: 1/24/2024
"""

import numpy as np
import matplotlib.pyplot as plt
from funcs.nesterov import NesterovFunctions
from opt.opt1 import gradient_descent, acc_gradient_descent, acc_gradient_descent2
from opt.opt2 import newton_plain, cubic_newton_slow, cubic_newton, acc_cubic_newton, newton_plain_no_param


def main():
    # Define Nesterov difficult tensor problem
    k = 10
    p = 2
    d = 25

    nesterov_problem = NesterovFunctions(k=k, p=p, d=d, use_gpu=True)

    start_theta = np.random.rand(d)
    start_theta /= np.linalg.norm(start_theta)
    start_theta = start_theta.astype(np.float32)
    print("THETA:")
    print(start_theta)

    # Run optimizers
    num_iters = 2000

    print("Performing gradient descent...")
    loss_o1, grad_o1, theta_star = gradient_descent(start_theta, num_iters, nesterov_problem)
    print("Performing accelerated gradient descent...")
    loss_o1_a, grad_a, theta_star = acc_gradient_descent(start_theta, num_iters, nesterov_problem)
    print("Performing accelerated gradient descent 2...")
    loss_o1_a2, grad_a2, theta_star = acc_gradient_descent(start_theta, num_iters, nesterov_problem)

    print("Performing Newton's method with quadratic regularization...")
    loss_o2_q, grad_q, theta_star = newton_plain(start_theta, num_iters, nesterov_problem, 1, 1)
    print("Performing Newton's method...")
    loss_o2_qp, grad_qp, theta_star = newton_plain_no_param(start_theta, num_iters, nesterov_problem)
    print("Performing Newton's method with cubic regularization...")
    loss_o2_c, grad_c, theta_star = cubic_newton_slow(start_theta, num_iters, nesterov_problem)
    print("Performing fast Newton's method with cubic regularization...")
    loss_o2_cf, grad_cf, theta_star = cubic_newton(start_theta, num_iters, nesterov_problem)
    print("Performing Accelerated Newton's method with cubic regularization...")
    loss_o2_acc, grad_acc, theta_star = acc_cubic_newton(start_theta, num_iters, nesterov_problem)

    # Plot loss over iterations
    print(loss_o2_acc)
    test1 = np.zeros((num_iters,), np.float32)
    test2 = np.zeros((num_iters,), np.float32)
    for k in range(len(test1)):
        test1[k] = 1
        test2[k] = 1

    print(theta_star)

    # plt.plot((loss_o1 - nesterov_problem.ideal_minimum()), label="Gradient Descent")  # - nesterov_problem.ideal_minimum()
    # plt.plot((loss_o1_a - nesterov_problem.ideal_minimum()), label="Acc. Gradient Descent")
    # plt.semilogy(loss_o1_a2 - nesterov_problem.ideal_minimum(), label="Acc. Gradient Descent 2")
    # plt.semilogy(loss_o2_q - nesterov_problem.ideal_minimum(), label="Quad. Newton")
    # # plt.plot(loss_o2_qp, label="Newton")
    plt.semilogy(loss_o2_c - nesterov_problem.ideal_minimum(), label="Cubic Newton")
    plt.semilogy(loss_o2_cf - nesterov_problem.ideal_minimum(), label="Fast Cubic Newton")
    plt.semilogy(loss_o2_acc - nesterov_problem.ideal_minimum(), label="Acc. Cubic Newton")
    plt.xlabel("Iterations")
    plt.ylabel("$f(x_k) - f^*$")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
