"""
Superclass for defining loss functions to be optimized with higher-order methods.
"""
from typing import Tuple

import numpy as np
from cuda.cuda import CUdevice_attribute, cuDeviceGetAttribute, cuDeviceGetName, cuInit


class HoopProblem:
    def __init__(self, use_gpu: bool):
        """

        :param use_gpu:
        """
        self.use_gpu = use_gpu

        if not use_gpu:
            return

        # The following code for getting GPU attributes is courtesy of Dr. Carlos Costa

        # Initialize CUDA Driver API
        (err,) = cuInit(0)

        # Get attributes of GPU starting with device name
        err, self.DEVICE_NAME = cuDeviceGetName(128, 0)
        self.DEVICE_NAME = self.DEVICE_NAME.decode("ascii").replace("\x00", "")

        err, self.MAX_THREADS_PER_BLOCK = cuDeviceGetAttribute(
            CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 0
        )  # maximum number of threads per block
        err, self.MAX_BLOCK_DIM_X = cuDeviceGetAttribute(
            CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, 0
        )  # maximum number of blocks per grid
        err, self.MAX_GRID_DIM_X = cuDeviceGetAttribute(
            CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, 0
        )  # maximum number of threads per grid
        err, self.SMs = cuDeviceGetAttribute(
            CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0
        )  # maximum number of multiprocessors

    def p_order_info(self, p: int, theta: np.array) -> Tuple:
        pass

    def lipschitz(self, p: int):
        pass
