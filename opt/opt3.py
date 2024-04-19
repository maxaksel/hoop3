"""
Contains all code for implementing Nesterov's third-order hyperfast optimization method.

Author: Max Bowman
Date: 12/22/2023
"""
import os
from typing import Tuple
import numpy as np
from funcs.problem import HoopProblem
from tqdm import tqdm
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt


def f_reg(theta: np.array, *args):
    start_theta, f_val, grads, H, T, M = args
    h = theta - start_theta
    val = f_val + h.T @ grads + 0.5 * h.T @ H @ h + (1/6) * (h.T @ T @ h) @ h
    val += (M/2) * np.linalg.norm(h) ** 4

    return val


def hyperfast_old(start_theta: np.array, num_iters: int, problem: HoopProblem) -> Tuple[np.array, np.array, np.array]:
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


def dual_auxiliary(tau: float, *args) -> float:
    gamma, T, c_tilda = args
    dot1 = np.linalg.lstsq(gamma*tau*np.eye(T.shape[0]) + np.diag(T), c_tilda, rcond=None)[0]
    return -0.5*tau*tau - 0.5*np.dot(dot1, c_tilda)

def auxiliary(tau: np.array, args) -> float:
    gamma, T, c_tilda = args
    dot1 = np.linalg.lstsq(gamma * tau * np.eye(T.shape[0]) + np.diag(T), c_tilda, rcond=None)[0]
    return -0.5 * tau * tau - 0.5 * np.dot(dot1, c_tilda)

def hyperfast(start_theta: np.array, num_iters: int, problem: HoopProblem) -> Tuple[np.array, np.array, np.array]:
    L3 = problem.lipschitz(3)
    M = 2*L3
    # gamma = tau*(1+tau)/2 * L3
    gamma = M/2
    # C = 1.5*np.sqrt(2*M*M - 2*L3*L3)
    # A_constant = ((M*M - L3*L3) / (8*M*M))**1.5 * 0.25**4
    theta_hat = np.copy(start_theta)

    loss = np.zeros((num_iters, 1), dtype=np.float32)
    grad_norms = np.zeros((num_iters, 1), dtype=np.float32)

    for k in tqdm(range(num_iters)):
        # Information-gathering
        f_val, grads, H, third_order_func = problem.p_order_info(3, theta_hat)
        loss[k] = f_val
        grad_norms[k] = np.linalg.norm(grads)

        # Iteratively solve the hyperfast auxiliary problem
        h = np.zeros((len(start_theta),), dtype=np.float32)

        for _ in range(10):  # change this
            c = grads + H @ h + 0.5 * third_order_func(h)
            Lambda, U = np.linalg.eig(H)
            c_tilda = U.T @ c

            # taus = np.linspace(0, 100, 10000)
            # vals = np.zeros((len(taus),), dtype=np.float32)
            # for i in range(len(taus)):
            #     vals[i] = auxiliary(taus[i], args=(gamma, Lambda, c_tilda))
            # plt.plot(taus, vals)
            # plt.show()

            # os.system('pause')
            best_tau = minimize_scalar(dual_auxiliary, args=(gamma, Lambda, c_tilda), options={'disp': False}).x
            best_tau = max(best_tau, 0.00001)
            # print(f"Best tau: {best_tau}")

            h = -U @ np.linalg.inv(gamma*best_tau*np.eye(Lambda.shape[0]) + Lambda) @ c_tilda
        print("Iter done.")

        # Update step
        theta_hat += h

    return loss, grad_norms, theta_hat
