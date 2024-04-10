# -*- coding: utf-8 -*-

"""
This module contains the problem definition for binary logistic regression.

Author: Max A. Bowman
Date: 2/21/2024
"""

from typing import Tuple
import numpy as np
import scipy
from overrides import overrides
from funcs.problem import HoopProblem
from numba import cuda
# from numba.cuda.cudadrv.devicearray import DeviceNDArray
from util.custom_kernels import fast_matmul, logistic_info, mat_vec
from time import perf_counter_ns


def sigma(x):
    return 1 / (1 + np.exp(-x))


"""
GPU Kernels for accelerating computation of logistic regression loss function, loss function gradient, and loss
function hessian can all be found below.
"""


@cuda.jit
def hessian_kernel(data_matrix, sigma_lookup, H):
    """
    CUDA Kernel for computing the Hessian.

    :param data_matrix:
    :param sigma_lookup:
    :param H:
    :return:
    """
    j, k = cuda.grid(2)

    # Bounds check
    if j >= data_matrix.shape[0] or k >= data_matrix.shape[1]:
        return

    if j <= k:
        for i in range(data_matrix.shape[0]):
            H[j, k] += data_matrix[i, j] * data_matrix[i, k] * sigma_lookup[i]
        H[k, j] = H[j, k]


class LogisticFunctions(HoopProblem):
    def __init__(self, data_matrix: np.array, labels: np.array, use_gpu: bool):
        """
        Initialize a new logistic regression problem.

        :param data_matrix:
        :param labels:
        :param use_gpu:
        """
        super().__init__(use_gpu)
        self.data_matrix = data_matrix
        self.labels = labels

        if use_gpu:
            self.dev_data_matrix = cuda.to_device(self.data_matrix)
            self.dev_labels = cuda.to_device(self.labels)
            self.threads_per_block = (8, 8)  # 64 threads per block is common

    def p_order_info_gpu(self, p: int, theta: np.array) -> Tuple:
        """

        :param p:
        :param theta:
        :return:
        """
        dev_theta = cuda.to_device(theta)

        if p > 2:
            raise Exception("Order p>2 not supported.")
        elif p < 0:
            raise Exception("Order p<0 not supported.")

        dev_logistic_residuals_array = cuda.device_array((self.data_matrix.shape[0],), dtype=np.float32)
        dev_labels_data_matrix_times_theta = cuda.device_array((self.data_matrix.shape[0],), dtype=np.float32)
        dev_data_matrix_times_theta = cuda.device_array((self.data_matrix.shape[0],), dtype=np.float32)
        dev_sigma_lookup = cuda.device_array((self.data_matrix.shape[0],), dtype=np.float32)
        dev_z = cuda.device_array((self.data_matrix.shape[0],), dtype=np.float32)

        block_size = 16*32
        logistic_info[block_size, 256](self.dev_data_matrix, dev_theta, self.dev_labels,
                                dev_data_matrix_times_theta, dev_labels_data_matrix_times_theta,
                                dev_sigma_lookup, dev_logistic_residuals_array, dev_z)

        residuals = dev_logistic_residuals_array.copy_to_host()

        # Compute Loss
        loss = np.sum(residuals) / self.data_matrix.shape[0]
        if p == 0:
            return loss

        # Compute Gradient
        dev_gradient = cuda.device_array((theta.shape[0],), dtype=np.float32)
        mat_vec[block_size, 256](self.dev_data_matrix.T, dev_z, dev_gradient)
        gradient = dev_gradient.copy_to_host()
        gradient /= self.data_matrix.shape[0]

        if p == 1:
            return loss, gradient

        # Compute Hessian
        dev_H = cuda.device_array((theta.shape[0], theta.shape[0]), dtype=np.float32)

        block_size = (dev_H.shape[0] // self.threads_per_block[0] + 1, dev_H.shape[1] // self.threads_per_block[1] + 1)
        # block_size = (10, 10)

        hessian_kernel[block_size, self.threads_per_block](self.dev_data_matrix, dev_sigma_lookup, dev_H)
        H = dev_H.copy_to_host()
        H /= self.data_matrix.shape[0]

        return loss, gradient, H  # if p == 2

    @overrides
    def p_order_info(self, p: int, theta: np.array) -> Tuple:
        """

        :param p:
        :param theta:
        :return:
        """
        if p > 3:
            raise Exception("Order p>3 not supported.")
        elif p < 0:
            raise Exception("Order p<0 not supported.")

        # if self.use_gpu:
        #     return self.p_order_info_gpu(p, theta)

        # CPU-based code
        data_matrix_times_theta = self.data_matrix @ theta
        labels_data_matrix_times_theta = self.labels * data_matrix_times_theta

        loss = np.sum(np.log(1 + np.exp(-labels_data_matrix_times_theta))) / self.data_matrix.shape[0]

        if p == 0:
            return loss

        z = -self.labels / (1 + np.exp(labels_data_matrix_times_theta))
        gradient = self.data_matrix.T @ z / self.data_matrix.shape[0]

        if p == 1:
            return loss, gradient

        H = np.zeros((len(theta), len(theta)))

        sigma_xw = sigma(data_matrix_times_theta)
        sigma_lookup = sigma_xw * (1 - sigma_xw)

        if not self.use_gpu:
            for j in range(len(theta)):
                for k in range(j, len(theta)):
                    H[j, k] = np.sum(self.data_matrix[:, j] * self.data_matrix[:, k] * sigma_lookup)
                    H[k, j] = H[j, k]
        else:
            dev_H = cuda.to_device(H)
            dev_sigma_lookup = cuda.to_device(sigma_lookup)

            block_size = (H.shape[0] // self.threads_per_block[0] + 1, H.shape[1] // self.threads_per_block[1] + 1)
            # block_size = (10, 10)

            hessian_kernel[block_size, self.threads_per_block](self.dev_data_matrix, dev_sigma_lookup, dev_H)
            cuda.synchronize()
            H = dev_H.copy_to_host()

        H /= self.data_matrix.shape[0]

        if p == 2:
            return loss, gradient, H  # if p == 2

        # Compute third-order tensor
        T = np.zeros((self.data_matrix.shape[1], self.data_matrix.shape[1], self.data_matrix.shape[1]), dtype=np.float32)

        third_order_lookup = self.labels * (np.exp(labels_data_matrix_times_theta) + np.exp(-labels_data_matrix_times_theta))
        third_order_lookup /= np.square(np.exp(labels_data_matrix_times_theta) - np.exp(-labels_data_matrix_times_theta))
        # print("LOOKUP size:")
        # print(third_order_lookup.shape)

        for j in range(self.data_matrix.shape[1]):
            for k in range(self.data_matrix.shape[1]):
                for l in range(self.data_matrix.shape[1]):
                    T[j, k, l] = 0.0
                    for i in range(self.data_matrix.shape[0]):
                        T[j, k, l] -= (1/self.data_matrix.shape[0]) * self.data_matrix[i, j] * self.data_matrix[i, k] \
                                        * self.data_matrix[i, l] * third_order_lookup[i]
        return loss, gradient, H, T

    @overrides
    def lipschitz(self, p: int) -> float:
        """
        max_sigma^2 is the largest eigenvalue of X'X.
        The first-order Lipschitz constant is bounded above by
        1/(4N) lambda(X'X), which is proportional to the
        induced two-norm of X'X.

        The second-order Lipschitz constant is bounded above by
        1/(3*sqrt(6)) * max(norm(x_i)) * sqrt(lambda(X'X)), which
        is proportional to the induced two-norm of X'X.

        :return: upper bound of the p-order Lipschitz constant.
        """
        max_sigma = scipy.linalg.svdvals(self.data_matrix)[0]
        if p == 1:
            return (1 / (4 * self.data_matrix.shape[0])) * max_sigma * max_sigma
        elif p == 2:
            return 1 / (6 * np.sqrt(3)) * (1 / self.data_matrix.shape[0]) * np.max(
                np.linalg.norm(self.data_matrix, axis=0)) * max_sigma  # TODO: Verify

        return -1
