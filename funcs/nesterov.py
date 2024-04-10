# -*- coding: utf-8 -*-

"""
This module contains the problem definition for Nesterov's difficult tensor methods.

Author: Max A. Bowman
Date: 1/05/2024
"""

import math
from typing import Tuple
import numpy as np
from overrides import overrides
from numba import cuda
from funcs.problem import HoopProblem


class NesterovFunctions(HoopProblem):
    def __init__(self, k: int, p: int, d: int, use_gpu: bool):
        """

        :param k:
        :param p:
        :param d:
        :param use_gpu:
        """
        super().__init__(use_gpu)

        self.k = k
        self.p = p
        self.d = d

        Uk = np.eye(k)
        for i in range(k - 2):
            Uk[i, i + 1] = -1

        Ink = np.eye(d - k)
        self.Ak = np.zeros((d, d), dtype=np.float32)

        self.Ak[0:k, 0:k] = Uk
        self.Ak[k:, k:] = Ink

        # print("Ak:")
        # print(self.Ak)

        if use_gpu:
            self.dev_Ak = cuda.to_device(self.Ak)

    def nu_p(self, x: np.array) -> np.array:
        """

        :param x:
        :return:
        """
        return 1 / (self.p + 1) * np.sum(np.power(np.abs(x), self.p + 1))

    @overrides
    def p_order_info(self, p: int, theta: np.array) -> Tuple:
        """

        :param p:
        :param theta:
        :return:
        """
        Akx = self.Ak @ theta
        val = self.nu_p(Akx) - theta[0]

        # print("Akx")
        # print(Akx)

        if p == 0:
            return val

        eta_gradient = np.power(np.abs(Akx), self.p - 1) * Akx
        gradient_final = self.Ak.T @ eta_gradient
        gradient_final[0] -= 1

        if p == 1:
            return val, gradient_final

        H = 3 * self.Ak.T @ np.diag(Akx) @ np.diag(Akx) @ self.Ak

        if p == 2:
            return val, gradient_final, H

    @overrides
    def lipschitz(self, p: int) -> float:
        """
        TODO: investigate
        :return:
        """

        # return 2 * 2 * math.factorial(p)
        return (2 ** 1.5) * math.factorial(p)

    def ideal_minimum(self) -> float:
        """

        :return:
        """
        return -self.k*self.p/(self.p+1)

    def ideal_params(self) -> np.array:
        """

        :return:
        """
        theta_solution = np.ones((self.d, 1))
        for i in range(0, self.d):
            val = self.k - (i + 1) + 1
            theta_solution[i] = val if val > 0 else 0

        return theta_solution
