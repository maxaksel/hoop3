"""
Contains all code for implementing Nesterov's third-order hyperfast optimization method.

Author: Max Bowman
Date: 12/22/2023
"""

from typing import Tuple
import numpy as np
from funcs.problem import HoopProblem
from tqdm import tqdm
from scipy.optimize import minimize


def f_reg(theta: np.array, *args):
    start_theta, f_val, grads, H, T, M = args
    h = theta - start_theta
    val = f_val + h.T @ grads + 0.5 * h.T @ H @ h + (1/6) * (h.T @ T @ h) @ h
    val += (M/2) * np.linalg.norm(h) ** 4

    return val


def hyperfast(start_theta: np.array, num_iters: int, problem: HoopProblem) -> Tuple[np.array, np.array, np.array]:
    L3 = problem.lipschitz(3)
    M = 2*L3
    # C = 1.5*np.sqrt(2*M*M - 2*L3*L3)
    # A_constant = ((M*M - L3*L3) / (8*M*M))**1.5 * 0.25**4
    theta_hat = np.copy(start_theta)

    loss = np.zeros((num_iters, 1), dtype=np.float32)
    grad_norms = np.zeros((num_iters, 1), dtype=np.float32)

    for k in tqdm(range(num_iters)):
        # Information-gathering
        f_val, grads, H, T = problem.p_order_info(3, theta_hat)
        loss[k] = f_val
        grad_norms[k] = np.linalg.norm(grads)
        # Update step
        theta_hat = minimize(f_reg, theta_hat, args=(theta_hat, f_val, grads, H, T, M)).x

    return loss, grad_norms, theta_hat
