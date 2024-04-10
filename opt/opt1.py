"""
Contains all code for implementing second-order optimization methods.

Author: Max Bowman
Date: 1/05/2024
"""

from typing import Tuple
import numpy as np
from tqdm import tqdm
from funcs.problem import HoopProblem


def gradient_descent(start_theta: np.array, num_iters: int, problem: HoopProblem) -> Tuple[np.array, np.array, np.array]:
    """

    :param start_theta:
    :param num_iters:
    :param problem:
    :return:
    """
    alpha = 1 / problem.lipschitz(1)
    print("Step size:")
    print(alpha)
    theta_hat = np.copy(start_theta)
    loss = np.zeros((int(num_iters),), dtype=np.float32)
    grad_norms = np.zeros((int(num_iters),), dtype=np.float32)

    for i in tqdm(range(int(num_iters))):
        loss[i], grad = problem.p_order_info(1, theta_hat)
        grad_norms[i] = np.linalg.norm(grad)
        theta_hat -= alpha * grad

    return loss, grad_norms, theta_hat


def acc_gradient_descent(start_theta: np.array, num_iters: int, problem: HoopProblem) -> Tuple[np.array, np.array, np.array]:
    alpha = 1 / problem.lipschitz(1)

    loss = np.zeros((int(num_iters),), dtype=np.float32)
    grad_norms = np.zeros((int(num_iters),), dtype=np.float32)
    theta_hat = np.copy(start_theta)  # x_{0}
    y = np.copy(start_theta)  # y_{1}
    t = 1  # t_{1}

    for i in tqdm(range(int(num_iters))):
        loss[i], grad_theta = problem.p_order_info(1, theta_hat)

        grad_norms[i] = np.linalg.norm(grad_theta)

        prev_theta = theta_hat
        theta_hat = y - alpha*grad_theta
        t_next = (1+np.sqrt(1+4*t*t))/2
        y = theta_hat + (t-1)/t_next*(theta_hat - prev_theta)  # update step

        t = t_next

    return loss, grad_norms, theta_hat


def acc_gradient_descent3(start_theta: np.array, num_iters: int, problem: HoopProblem) -> Tuple[np.array, np.array]:
    """

    :param start_theta:
    :param num_iters:
    :param problem:
    :return:
    """
    alpha = 1 / problem.lipschitz(1)

    lam = np.zeros((int(num_iters + 1),), dtype=np.float32)
    loss = np.zeros((int(num_iters),), dtype=np.float32)
    theta_hat = np.copy(start_theta)
    y_prev = np.copy(start_theta)

    for i in tqdm(range(1, int(num_iters + 1))):
        lam[i] = 0.5 * (1 + np.sqrt(1 + 4 * lam[i - 1] * lam[i - 1]))

    for i in tqdm(range(int(num_iters))):
        loss[i], grads = problem.p_order_info(1, theta_hat)

        gamma = (1 - lam[i]) / (lam[i + 1])
        y_curr = theta_hat - alpha * grads
        theta_hat = (1 - gamma) * y_curr + gamma * y_prev
        y_prev = y_curr

    return loss, theta_hat


def acc_gradient_descent2(start_theta: np.array, num_iters: int, problem: HoopProblem) -> Tuple[np.array, np.array]:
    """
    Accelerated gradient descent as described in INDE 517.

    :param start_theta: initial guess.
    :param num_iters: number of iterations before stopping.
    :param problem: HoopProblem instance to solve.
    :return: a tuple with loss followed by the problem solution.
    """
    alpha = 1 / problem.lipschitz(1)
    beta = 0
    rho = 0

    loss = np.zeros((int(num_iters),), dtype=np.float32)
    theta_hat = np.copy(start_theta)
    theta_prev = np.copy(start_theta)

    for i in tqdm(range(int(num_iters))):
        loss[i] = problem.p_order_info(0, theta_hat)

        y = theta_hat + beta*(theta_hat - theta_prev)

        _, grads = problem.p_order_info(1, y)
        theta_prev = theta_hat
        theta_hat = y - alpha*grads

        beta = rho**2
        rho = 0.5*((rho**2 - 1) + np.sqrt((rho**2 - 1)**2 + 4))
        beta *= rho  # beta_{k+1} = rho_{k}^2 * rho_{k+1}

    return loss, theta_hat
