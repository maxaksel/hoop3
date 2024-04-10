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
from util.custom_kernels import fast_matmul
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


class OptimalTransport(HoopProblem):
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

        self.M = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                self.M[i, j] = (i - j)*(i - j)/(n*n)  # normalize by matrix size to stabilize numerically

        self.A = np.zeros((2*n, n*n), dtype=np.float32)
        for k in range(n):
            self.A[0:n, k * n:k * n + n] = np.eye(n)
        for i in range(n, 2 * n):
            self.A[i, (i - n) * n:(i - n + 1) * n] = 1.0

        self.At = np.transpose(self.A)
        self.vecM = np.ndarray.flatten(self.M)

        self.La = scipy.linalg.svdvals(self.A).max()

        if self.use_gpu:
            self.devA = cuda.to_device(self.A)
            self.devAt = cuda.to_device(self.At)
            self.devVecM = cuda.to_device(self.vecM)
            self.devM = cuda.to_device(self.M)

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

        smax_in = np.matmul(self.At, theta) - self.vecM
        loss = softmax(self.gamma, smax_in) - np.dot(theta, self.b)

        if p == 0:
            return loss

        grads = self.A @ grad_smax(self.gamma, self.At @ theta - self.vecM) - self.b

        if p == 1:
            return loss, grads

        if not self.use_gpu:
            H = self.A @ hessian_smax(self.gamma, self.At @ theta - self.vecM) @ self.At
        else:
            devH = cuda.device_array((len(theta), len(theta)))
            smaxH = hessian_smax(self.gamma, self.At @ theta - self.vecM)
            devSmaxH = cuda.to_device(smaxH)
            fast_matmul[(128, 128), 16](devSmaxH, self.devAt, devH)
            # print("HERE")
            # fast_matmul[(16, 16), 128](self.devA, devH, devH)
            H = devH.copy_to_host()

        if p == 2:
            return loss, grads, H

    def wasserstein_distance(self, opt_lam: np.array) -> Tuple[np.array, np.array]:
        """
        Computes the optimal transport plan and entropy-regularized Wasserstein distance for optimal ksi and eta
        parameters of a particular pair of distributions.

        :param opt_lam: optimal ksi and eta parameters for a particular pair of distributions obtained by running some
        minimizer on loss_func, grad_func, and hessian_func.
        :return: a tuple containing a Kantorovich transport plan followed by the entropy-regularized Wasserstein
        distance.
        """
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
        # return 10
        return self.La * ((p+1)/np.log(p+2))**(p+1)*math.factorial(p)/np.power(self.gamma, p)
