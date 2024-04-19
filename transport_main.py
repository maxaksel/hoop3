# -*- coding: utf-8 -*-

"""
Optimal transport application demonstration.

Author: Max A. Bowman
Date: 1/11/2024
"""

import numpy as np
import scipy
from funcs.transport import OptimalTransport
from opt.opt1 import gradient_descent, acc_gradient_descent
from opt.opt2 import newton_plain, cubic_newton_slow, cubic_newton, acc_cubic_newton
from opt.opt3 import hyperfast
import matplotlib.pyplot as plt


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    n = 25
    gamma = 0.1

    transport_problem = OptimalTransport(n=n, gamma=gamma, use_gpu=False)

    for p in range(1, 4):
        print(f"{p}-order Lipschitz: {transport_problem.lipschitz(p)}")

    p = np.random.normal(loc=np.zeros(n,), scale=np.ones(n,)).astype(np.float32)
    p = (p - np.min(p))
    p[int(n / 2):] = 0
    p /= p.sum()
    q = np.random.normal(loc=np.zeros(n,), scale=np.ones(n,)).astype(np.float32)
    q = (q - np.min(q))
    q[:int(n / 2)] = 0
    q /= q.sum()

    # Test simpler discrete case
    # epsilon = 1e-5
    # p = np.array([0, 0, 1-10*epsilon, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32) + epsilon
    # q = np.array([0, 0.5-5*epsilon, 0, 0, 0, 0, 0, 0, 0, 0.5-5*epsilon]).astype(np.float32) + epsilon

    # Test simpler discrete case for n = 100
    # epsilon = 1e-5
    # p[0:] = epsilon
    # q[0:] = epsilon
    # p[0] = 1 - 99*epsilon
    # q[-1] = 1 - 99*epsilon

    plt.figure()
    plt.stem(p)
    plt.show()

    plt.figure()
    plt.stem(q)
    plt.show()

    transport_problem.set_distributions(p, q)
    #
    # print(np.max(scipy.linalg.eigvals(transport_problem.A @ transport_problem.At)))
    # print(np.min(scipy.linalg.eigvals(transport_problem.A @ transport_problem.At)))
    # print(transport_problem.A)
    # print(transport_problem.At)
    #
    # plt.stem(scipy.linalg.eigvals(transport_problem.A @ transport_problem.At))
    # plt.show()
    #
    num_iters = 50

    # Gradient Descent
    theta_hat = np.random.rand(2 * n)
    orig_theta_hat = np.copy(theta_hat)
    print("Performing gradient descent...")
    loss_gd, grad_gd, theta_i = gradient_descent(theta_hat, num_iters, transport_problem)

    # Accelerated Gradient Descent
    theta_hat = np.copy(orig_theta_hat)
    print("Performing accelerated gradient descent...")
    loss_agd, grad_agd, theta_agd = acc_gradient_descent(theta_hat, num_iters, transport_problem)

    # Regular Newton's Method
    theta_hat = np.copy(orig_theta_hat)
    print("Performing Newton's method with quadratic regularization...")
    loss_newton, grad_newton, theta_n = newton_plain(theta_hat, num_iters, transport_problem, alpha=1, gamma=1)

    # Cubic Regularization Slow
    theta_hat = np.copy(orig_theta_hat)
    print("Performing Newton's method with cubic regularization (slow)...")
    loss_na2, grad_na2, theta_ncs = cubic_newton_slow(theta_hat, num_iters, transport_problem)

    # Cubic Regularization Fast
    theta_hat = np.copy(orig_theta_hat)
    print("Performing Newton's method with cubic regularization (fast)...")
    loss_a2, grad_a2, theta_ncf = cubic_newton(theta_hat, num_iters, transport_problem)

    # Hyperfast Method
    # theta_hat = np.copy(orig_theta_hat)
    # print("Performing Hyperfast method")
    # loss_3, grad_3, theta_3 = hyperfast(theta_hat, num_iters, transport_problem)

    # Accelerated Cubic Newton
    theta_hat = np.copy(orig_theta_hat)
    print("Performing accelerated Newton's method with cubic regularization...")
    loss_acc2, grad_acc2, theta_acf = acc_cubic_newton(theta_hat, num_iters, transport_problem)

    plt.semilogy(loss_gd, label="Gradient Descent")
    plt.semilogy(loss_agd, label="Acc. Gradient Descent")
    plt.semilogy(loss_newton, label="Quadratic Newton")
    plt.semilogy(loss_na2, label="Cubic Reg. Newton Method (Slow)")
    plt.semilogy(loss_a2, label="Cubic Reg. Newton Method (Fast)")
    # plt.plot(loss_3, label="Hyperfast)")
    plt.semilogy(loss_acc2, label="Acc. Cubic Reg. Newton Method")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.semilogy(grad_gd, label="Gradient Descent")
    plt.semilogy(grad_agd, label="Acc. Gradient Descent")
    plt.semilogy(grad_newton, label="Quadratic Newton")
    plt.semilogy(grad_na2, label="Cubic Reg. Newton Method (Slow)")
    plt.semilogy(grad_a2, label="Cubic Reg. Newton Method (Fast)")
    # plt.plot(loss_3, label="Hyperfast)")
    plt.semilogy(grad_acc2, label="Acc. Cubic Reg. Newton Method")
    plt.xlabel("Iterations")
    plt.ylabel("Gradient Norm")
    plt.legend()
    plt.show()

    # zero_transport = np.zeros((2*n,), dtype=np.float32)
    T, dist = transport_problem.wasserstein_distance(theta_agd)
    # T, dist = transport_problem.wasserstein_distance(zero_transport)
    # print("Theta star")
    # print(theta_star)
    # print("Transport plan:")
    # print(T)
    plt.imshow(T)
    plt.title("2-Wasserstein Distance: " + str(dist.item()))
    plt.show()
