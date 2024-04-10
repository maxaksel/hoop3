# -*- coding: utf-8 -*-

"""
This module contains general GPU kernels for functions like matrix multiplication or matrix-vector
multiplication. The fast_matmul kernel is from https://numba.pydata.org/numba-doc/dev/cuda/examples.html.

"""

import math
import numpy as np
from numba import cuda, float32
# from numba.cuda.cudadrv.devicearray import DeviceNDArray


# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
TPB = 16


@cuda.jit
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x  # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp


@cuda.jit
def mat_vec(A, x, Ax):
    """
    Performs matrix-vector multiplication.

    :param A: matrix
    :param x: vector
    :param Ax: matrix product A@x
    :return: None
    """
    i = cuda.grid(1)

    if i >= A.shape[0]:
        return

    Ax[i] = 0.0
    for k in range(A.shape[1]):
        Ax[i] += A[i, k] * x[k]


@cuda.jit
def logistic_info(A, x, y,
                  Ax, yAx,
                  sigma_lookup, residuals, z):
    """
    GPU Kernel for computing various quantities important for logistic regression.

    :param A: data matrix (in)
    :param x: parameters (in)
    :param y: labels (in)
    :param Ax: product A@x (out)
    :param yAx: y' .* A@x (out)
    :param sigma_lookup: used for computing the gradient of logistic loss (out)
    :param residuals: the sum of these is logistic loss (out)
    :param z: used for computing the gradient of logistic regression (out)
    :return: None
    """
    start_i = cuda.grid(1)
    threads_per_grid = cuda.gridsize(1)

    # if i == 1:
    #     print("HERE:")
    #     print(x.shape[0])
    #     from pdb import set_trace
    #     set_trace()  # from CUDA website

    for i in range(start_i, A.shape[0], threads_per_grid):
        Ax[i] = 0.0
        for k in range(A.shape[1]):
            Ax[i] += A[i, k]*x[k]

        yAx[i] = Ax[i]*y[i]
        residuals[i] = math.log(1 + math.exp(yAx[i]))

        sigma_xw = 1 / (1 + math.exp(-Ax[i]))
        sigma_lookup[i] = sigma_xw * (1 - sigma_xw)

        z[i] = -y[i] / (1 + math.exp(yAx[i]))
