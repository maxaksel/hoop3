# -*- coding: utf-8 -*-

"""
Definition of the 2-Wasserstein metric computation problem.
"""
import math
from typing import Tuple
import numpy as np
import scipy
from overrides import overrides
from funcs.problem import HoopProblem
from numba import cuda
from time import perf_counter_ns


def softmax(gamma, in_vector):
    return gamma*scipy.special.logsumexp(in_vector/gamma, 0)


def grad_smax(gamma, x: np.array) -> np.array:
    # print("Lambda:")
    # print(x)
    scaled_x = x/gamma
    scaled_x -= scaled_x.max()
    exp_vector = np.exp(scaled_x)
    # print("Exp vector:")
    # print(exp_vector)
    # print(exp_vector.max())
    # print(np.sum(exp_vector))
    result = exp_vector / np.sum(exp_vector)
    # print(result)

    return result


def hessian_smax(gamma, x: np.array) -> np.array:
    grad = grad_smax(gamma, x)
    # diag_grad = (np.diag(grad))
    # outer_product = (np.outer(grad, grad))
    return 1/gamma * (np.diag(grad) - np.outer(grad, grad))


class OptimalTransport2(HoopProblem):
    def __init__(self, n: int, gamma: float, use_gpu: bool):
        """

        :param n:
        :param gamma:
        :param use_gpu:
        """
        super().__init__(use_gpu)

        self.n = n
        self.gamma = gamma
        self.use_gpu = use_gpu
        self.p = None
        self.q = None
        self.b = None

        self.M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                self.M[i, j] = (i - j)*(i - j)/(n*n)  # normalize by matrix size to stabilize numerically

        self.A = np.zeros((2*n, n*n))
        for k in range(n):
            self.A[0:n, k * n:k * n + n] = np.eye(n)
        for i in range(n, 2 * n):
            self.A[i, (i - n) * n:(i - n + 1) * n] = 1.0

        self.At = np.transpose(self.A)
        self.vecM = np.ndarray.flatten(self.M)
        self.LLS = np.linalg.pinv(self.At)  # numerically stable enough?
        self.LLSt = np.transpose(self.LLS)
        self.add = self.LLS @ self.vecM

    def set_distributions(self, p: np.array, q: np.array):
        """

        :param p: column vector
        :param q: column vector
        :return:
        """
        self.p = p
        self.q = q
        self.b = np.hstack((p, q))

    @overrides
    def p_order_info(self, p: int, theta: np.array) -> Tuple:
        """
        Returns higher-order derivatives up to the p-th order.

        :param p: order of interest (generally 1, 2, or 3).
        :param theta: generally referred to as lambda in the literature, this is xi stacked on eta.
        :return: a tuple with the loss, gradient, Hessian, etc.
        """
        if p < 0 or p > 2:
            raise Exception("0 <= p <= 2 is necessary.")

        lam = self.LLS @ theta + self.add
        # print("LAM")
        # print(lam)
        loss = softmax(self.gamma, theta) - np.dot(lam, self.b)

        if p == 0:
            return loss

        grads = grad_smax(self.gamma, theta) - self.LLSt @ self.b  # (A'A)A'... to recover what should be here instead of self.b

        if p == 1:
            return loss, grads

        H = hessian_smax(self.gamma, theta)

        if p == 2:
            return loss, grads, H

    def wasserstein_distance(self, opt_y: np.array) -> Tuple[np.array, np.array]:
        """
        Computes the optimal transport plan and entropy-regularized Wasserstein distance for optimal ksi and eta
        parameters of a particular pair of distributions.

        :param opt_y: optimal conjugate y parameter.
        :return: a tuple containing a Kantorovich transport plan followed by the entropy-regularized Wasserstein
        distance.
        """
        opt_lam = self.LLS @ (opt_y + self.vecM)
        xi = opt_lam[0:self.n]
        eta = opt_lam[self.n:]

        M_adjust = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                M_adjust[i, j] = (-self.M[i, j] + xi[i] + eta[j]) / self.gamma
        M_adjust -= np.max(M_adjust)

        transport_plan = np.exp(M_adjust) / np.exp(M_adjust).sum()

        # entropy = -np.sum(np.log(transport_plan + 1e-14) * transport_plan)
        # wasserstein_distance = np.sum(self.M * transport_plan) - self.gamma * entropy
        wasserstein_distance = self.n*self.n*np.sum(self.M * transport_plan)

        return transport_plan, wasserstein_distance

    @overrides
    def lipschitz(self, p: int) -> float:
        """
        TODO: fix this
        :param p:
        :return:
        """
        # L_mat = (p+1)/(np.log(p+2))**(p+1) * math.factorial(p) / self.gamma**p
        # print("L_mat:")
        # print(L_mat)
        # # numerator = ((p+1)/(np.log(p+2)))**(p+1)*math.factorial(p)
        # # denominator = self.gamma ** p
        # # return numerator/denominator
        return 10
        # return 4 * ((p+1)/np.log(p+2))**(p+1)*math.factorial(p)/np.power(self.gamma, p)
