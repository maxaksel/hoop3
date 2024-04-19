# -*- coding: utf-8 -*-

"""
Contains all code for implementing second-order optimization methods.

Author: Max Bowman
Date: 3/06/2024
"""

from typing import Tuple, Optional
import numpy as np
from scipy.optimize import fsolve, newton
from tqdm import tqdm
from funcs.problem import HoopProblem


def newton_plain_no_param(start_theta: np.array, num_iters: int, problem: HoopProblem) -> Tuple[np.array, np.array, np.array]:
    """
    Implementation of Newton's method with two regularization parameters (alpha and gamma).

    :param start_theta: starting guess for Newton's method.
    :param num_iters: number of iterations to run Newton's method.
    :param problem: a HoopProblem object defining the function to minimize.
    :return: a tuple (loss over time, final theta vector).
    """
    loss = np.zeros((int(num_iters),))
    grad_norms = np.zeros((int(num_iters),), dtype=np.float32)
    theta_hat = np.copy(start_theta)

    for i in tqdm(range(int(num_iters))):
        loss[i], grads, H = problem.p_order_info(2, theta_hat)
        grad_norms[i] = np.linalg.norm(grads)

        print(np.linalg.cond(H))
        h = -np.linalg.inv(H) @ grads
        # h = np.linalg.lstsq(-H, grads, rcond=None)[0]  # lstsq is faster than solve for large systems
        theta_hat += np.real(h)

    return loss, grad_norms, theta_hat


def newton_plain(start_theta: np.array, num_iters: int, problem: HoopProblem,
                 alpha: float, gamma: float) -> Tuple[np.array, np.array, np.array]:
    """
    Implementation of Newton's method with two regularization parameters (alpha and gamma).

    :param start_theta: starting guess for Newton's method.
    :param num_iters: number of iterations to run Newton's method.
    :param problem: a HoopProblem object defining the function to minimize.
    :param alpha: used to stabilize ill-conditioned Hessian matrices.
    :param gamma: should be in range [0, 1].
    :return: a tuple (loss over time, final theta vector).
    """
    loss = np.zeros((int(num_iters),))
    grad_norms = np.zeros((int(num_iters),), dtype=np.float32)
    theta_hat = np.copy(start_theta)

    for i in tqdm(range(int(num_iters))):
        loss[i], grads, H = problem.p_order_info(2, theta_hat)
        grad_norms[i] = np.linalg.norm(grads)

        h = np.linalg.lstsq(-(H + (1.0/alpha)*np.eye(H.shape[0]))/gamma,
                            grads, rcond=None)[0]  # lstsq is faster than solve for large systems

        theta_hat += np.real(h)

    return loss, grad_norms, theta_hat


def grad_f_reg(theta: np.array, *args) -> float:
    """
    Gradient of regularized subproblem to solve.
    """
    theta_orig, grads, H, M = args
    return grads + H@(theta-theta_orig) + (M/2)*np.linalg.norm(theta-theta_orig)*(theta-theta_orig)


def grad_f_reg_step(h: np.array, grads, H, M) -> float:
    """
    Gradient of regularized subproblem to solve.
    """
    return grads + H@h + (M/2)*np.linalg.norm(h) * h


def cubic_newton_slow(start_theta: np.array, num_iters: int, problem: HoopProblem) -> Tuple[np.array, np.array, np.array]:
    """
    Implementation of Nesterov's cubic Newton method without using the single-variable dual problem to reduce
    computational complexity.

    :param start_theta: initial guess for optimal point in feature space.
    :param num_iters: number of iterations to run as termination condition.
    :param problem: a HoopProblem description of the function you would like to minimize.
    :return: a tuple containing an array of loss over iterations followed by the minimizer.
    """
    loss = np.zeros((int(num_iters),), dtype=np.float32)
    grad_norms = np.zeros((int(num_iters),), dtype=np.float32)
    theta_hat = np.copy(start_theta)
    M = problem.lipschitz(2)
    d = len(start_theta)

    for i in tqdm(range(int(num_iters))):
        loss[i], grads, H = problem.p_order_info(2, theta_hat)
        grad_norms[i] = np.linalg.norm(grads)
        theta_hat = fsolve(grad_f_reg, np.zeros(d), args=(theta_hat, grads, H, M))

    return loss, grad_norms, theta_hat


def opt_subproblem_a(r, *args):
    """
    Auxiliary single-dimensional problem to solve as part of the accelerated cubic Newton algorithm.

    :param r: a floating point number.
    :param args: a tuple with eigenvalue diagonal matrix of Hessian, U^T @ grads, and the second-order Lipschitz
    constant.
    :return: value of the optimization subproblem.
    """
    Lambda, g_bar, M = args

    result = 0.0
    for i in range(Lambda.shape[0]):
        result += g_bar[i]**2 / (Lambda[i] + M*r/2)**2

    return result - r*r


def opt_subproblem_a_prime(r, *args):
    """
    Derivative of auxiliary problem.

    """
    Lambda, g_bar, M = args
    result = 0.0

    for i in range(Lambda.shape[0]):
        result += -M*g_bar[i]*g_bar[i] / (Lambda[i] + M*r/2)**3

    return result - 2*r


def opt_subproblem_a_prime2(r, *args):
    """
    Second derivative of auxiliary problem.

    """
    Lambda, g_bar, M = args
    result = 0.0

    for i in range(Lambda.shape[0]):
        result += 1.5*M*M*g_bar[i]*g_bar[i] / (Lambda[i] + M*r/2)**4

    return result - 2


def opt_subproblem(r, Lambda, g_bar, M):
    """
    Auxiliary single-dimensional problem to solve as part of the accelerated cubic Newton algorithm.

    :param r: a floating point number.
    :param args: a tuple with eigenvalue diagonal matrix of Hessian, U^T @ grads, and the second-order Lipschitz
    constant.
    :return: value of the optimization subproblem.
    """

    result = 0.0
    for i in range(Lambda.shape[0]):
        result += g_bar[i]**2 / (Lambda[i] + M*r/2)**2

    return result - r*r


def opt_subproblem_prime(r, Lambda, g_bar, M):
    """
    Derivative of auxiliary problem.

    """
    result = 0.0

    for i in range(Lambda.shape[0]):
        result += -M*g_bar[i]*g_bar[i] / (Lambda[i] + M*r/2)**3

    return result - 2*r


def opt_subproblem_prime2(r, Lambda, g_bar, M):
    """
    Second derivative of auxiliary problem.

    """
    result = 0.0

    for i in range(Lambda.shape[0]):
        result += 1.5*M*M*g_bar[i]*g_bar[i] / (Lambda[i] + M*r/2)**4

    return result - 2


def halley(start_r, epsilon, Lambda, g_bar, M):
    r = start_r

    while abs(opt_subproblem(r, Lambda, g_bar, M)) > epsilon:
        f_eval = opt_subproblem(r, Lambda, g_bar, M)
        fp_eval  = opt_subproblem_prime(r, Lambda, g_bar, M)
        fp2_eval = opt_subproblem_prime2(r, Lambda, g_bar, M)

        r -= 2*f_eval*fp_eval/(2*fp_eval*fp_eval - f_eval*fp2_eval)

    return r


def cubic_newton(start_theta: np.array, num_iters: int, problem: HoopProblem,
                 M: Optional[float] = None) -> Tuple[np.array, np.array, np.array]:
    """
    Implementation of Nesterov's cubic Newton method.

    :param start_theta: initial guess for optimal point in feature space.
    :param num_iters: number of iterations to run as termination condition.
    :param problem: a HoopProblem description of the function you would like to minimize.
    :param M: optional for different step sizes.
    :return: a tuple containing an array of loss over iterations followed by the minimizer.
    """
    loss = np.zeros((int(num_iters),), dtype=np.float32)
    grad_norms = np.zeros((int(num_iters),), dtype=np.float32)
    theta_hat = np.copy(start_theta)

    if M is None:
        M = problem.lipschitz(2)  # Hessian Lipschitz

    for i in tqdm(range(int(num_iters))):
        loss[i], grads, H = problem.p_order_info(2, theta_hat)
        grad_norms[i] = np.linalg.norm(grads)
        Lambda, U = np.linalg.eig(H)
        U = np.real(U)  # U should be real-valued
        Lambda = np.real(Lambda)

        g_bar = U.T@grads

        # r_star = halley(0.5, 1e-8, Lambda, g_bar, M)
        r_star = newton(opt_subproblem_a, 0.5, fprime=opt_subproblem_a_prime,
                        fprime2=opt_subproblem_a_prime2, args=(Lambda, g_bar, M))

        # print(f"r_star: {r_star}")
        if r_star < 0:
            raise Exception(f"r < 0!     r={r_star}")

        h = np.linalg.lstsq((np.diag(Lambda) + M*r_star/2*np.eye(H.shape[0]))@U.T, -g_bar, rcond=None)[0]
        # h = -U@np.linalg.inv(np.diag(Lambda) + M*r_star/2*np.eye(H.shape[0]))@g_bar
        theta_hat += np.real(h)

    return loss, grad_norms, theta_hat


def acc_cubic_newton(start_theta: np.array, num_iters: int, problem: HoopProblem) -> Tuple[np.array, np.array, np.array]:
    """
    Implementation of Nesterov's accelerated cubic Newton method.

    :param start_theta: initial guess for optimal point in feature space.
    :param num_iters: number of iterations to run as termination condition.
    :param problem: a HoopProblem description of the function you would like to minimize.
    :return: a tuple containing an array of loss over iterations followed by the minimizer.
    """
    loss = np.zeros((int(num_iters),), dtype=np.float32)
    loss_grad = np.zeros((int(num_iters),), dtype=np.float32)
    L3 = problem.lipschitz(2) * 5
    M = 2*L3
    N = 12*L3

    loss[0], loss_grad_0 = problem.p_order_info(1, start_theta)
    loss_grad[0] = np.linalg.norm(loss_grad_0)
    _, _, theta_hat = cubic_newton(start_theta, 1, problem, L3)  # i=0 -> i=1 (first iteration)
    l_grad = np.zeros((len(theta_hat),), dtype=np.float32)

    for k in tqdm(range(1, int(num_iters))):
        loss[k], x_grad = problem.p_order_info(1, theta_hat)  # get grad(x_k)
        loss_grad[k] = np.linalg.norm(x_grad)
        l_grad += k*(k+1)/2*x_grad  # get grad(l_k)

        v = start_theta - np.sqrt(2/N) * l_grad/np.sqrt(np.linalg.norm(l_grad))  # v_k
        y = k/(k+3) * theta_hat + 1/(k+3) * v  # y_k

        # Compute next x iterate from y_k
        _, y_grad, H = problem.p_order_info(2, y)

        Lambda, U = np.linalg.eig(H)
        U = np.real(U)  # U should be real-valued
        Lambda = np.real(Lambda)

        g_bar = U.T @ y_grad

        # r_star = halley(0.5, 1e-8, Lambda, g_bar, M)
        r_star = newton(opt_subproblem_a, 0.5, tol=1e-15, fprime=opt_subproblem_a_prime,
                        fprime2=opt_subproblem_a_prime2, args=(Lambda, g_bar, M))

        if r_star < 0:
            raise Exception(f"r < 0!     r={r_star}")

        # h_slow = fsolve(grad_f_reg, np.zeros(len(start_theta)), args=(theta_hat, y_grad, H, M))
        # print(f"H_SLOW: {h_slow}")
        # h_slow = next_theta_hat - theta_hat
        h2 = np.linalg.lstsq((np.diag(Lambda) + M * r_star / 2 * np.eye(H.shape[0])) @ U.T, -g_bar, rcond=None)[0]
        # h = -U@np.linalg.inv(np.diag(Lambda) + M*r_star/2*np.eye(H.shape[0]))@g_bar

        # print("DIFF:")
        # print(np.linalg.norm(h2 - h_slow))
        # # theta_hat = fsolve(grad_f_reg, np.zeros(len(theta_hat),), args=(theta_hat, y_grad, H, M))
        # print("grad_f_reg:")
        _, grads, H = problem.p_order_info(2, theta_hat)
        # print(np.linalg.norm(grad_f_reg_step(np.real(h_slow), grads, H, M)))

        theta_hat += np.real(h2)

    return loss, loss_grad, theta_hat


def grad_psi(x: np.array, *args) -> np.array:
    A, grad_list, x0 = args
    t = len(grad_list)

    final_grad = np.linalg.norm(x - x0) * (x - x0)
    for i in range(t):
        final_grad += (A[i+1] - A[i]) * grad_list[i]

    return final_grad


def acc_cubic_newton2(start_theta: np.array, num_iters: int, problem: HoopProblem) -> Tuple[np.array, np.array]:
    """
    Implementation of Nesterov's accelerated cubic Newton method.

    :param start_theta: initial guess for optimal point in feature space.
    :param num_iters: number of iterations to run as termination condition.
    :param problem: a HoopProblem description of the function you would like to minimize.
    :return: a tuple containing an array of loss over iterations followed by the minimizer.
    """

    M = 2*problem.lipschitz(2)
    # M = problem.lipschitz(2)
    d = len(start_theta)
    grad_list = []
    theta_0 = np.copy(start_theta)

    loss = np.zeros((num_iters,), dtype=np.float32)

    Ak = np.zeros((num_iters+1,))
    for k in range(len(Ak)):
        Ak[k] = (1/(2*M)) * ((k/3) ** 3)

    # print(" A")
    # print(Ak)

    v = np.copy(start_theta)
    theta_hat = np.copy(start_theta)

    loss[0], grad_theta0 = problem.p_order_info(1, start_theta)

    for t in tqdm(range(num_iters - 1)):
        y = v + Ak[t]/Ak[t+1]*(theta_hat - v)

        _, grads, H = problem.p_order_info(2, y)
        theta_hat = fsolve(grad_f_reg, np.zeros(d), args=(y, grads, H, M))
        loss[t+1], grad_theta = problem.p_order_info(1, theta_hat)
        grad_list.append(grad_theta)
        v = fsolve(grad_psi, np.zeros(d), args=(Ak, grad_list, theta_0))
        # print("V")
        # print(v)

    return loss, theta_hat
