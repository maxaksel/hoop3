# Hoop3

### Overview

Hoop3 is a Python library designed to make research in higher-order optimization
easier. The library is designed to be extensible to suit your research needs and includes
support for GPU acceleration. Please see the `docs` folder for more detailed descriptions
of the methods implemented and the performance of the library. The following methods
are implemented:

1. Gradient Descent
2. Accelerated Gradient Descent
3. Newton's method (damped and quadratically regularized)
4. Newton's method (cubic regularized)
5. Accelerated Newton's method (cubic regularized)
6. Nesterov's Hyperfast (Third-Order) Tensor method

### Installation instructions

To use this software, please make sure you have Python installed on your machine.
Then, to install all necessary dependencies, please run the following command
from the root directory:
`pip install requirements.txt`.

Then, any of the driver files can be run. For example, from the root
directory, you may run `python libsvm_main.py`.

### Intended Workflow

If you have a higher-order smooth problem you would like to solve using Hoop3,
add a module to the `funcs` folder. For example, let's say you create a file called
`funcs/test.py`. Within this file, you must define a class that extends the `HoopProblem`
class in `funcs/problem.py`. The only two methods you must implement in this class
are `p_order_info` and `lipschitz`. The first of these methods returns zero-, up to first-, second-,
and third-order information about the function of interest at a particular iterate. For zero- through second-order, this would
be the function value, function gradient, and function Hessian. Third-order information is implemented
slightly differently. Instead of returning a third-order tensor, the method should
return a Python Callable that takes a direction `h` and returns the third-order
tensor in the direction of `[h]^2` (in other words, a function computing
`D^3 f(x) [h]^2`). The second of these methods returns the `p`-order Lipschitz constant of the
particular application for the appropriate determination of step sizes. Please note that your Lipschitz constants
should ideally be tight to improve performance.

### Supported Problem Classes

By default, Hoop3 supports three problem classes. The first is optimal transport, with
a particular focus on computing the 2-Wasserstein distance. Please see `transport_main.py`
for an example of this functionality. Hoop3 also supports solving binary logistic regression
problems (with labels being either -1 or +1). Please see `logistic_main.py` and
`libsvm_main.py` for examples using both synthetic data and real data. Finally, Hoop3 supports
the minimization of Nesterov's difficult tensor functions, which are essentially test functions
constructed to be difficult for the optimization methods implemented in this library. They have analytical
solutions, so the primary reason for their inclusion is as a test of method correctness.

### GPU Acceleration
GPU acceleration was implemented for computing the logistic regression Hessian. It resulted in
at least a 6x speedup for all supported second-order methods as measured on 25-dimensional synthetic
data. The current GPU-acceleration strategy for second-order information is to assign each GPU thread
to compute each Hessian entry.

### Future Directions

The primary next direction for this library is implementing adaptive step sizes. We believe
this could fix convergence issues with accelerated cubic Newton method and the hyperfast
method. Another future direction is the GPU acceleration of the optimal transport application.
Considering much of the computation for this application is matrix-matrix and matrix-vector multiplication,
we believe the cuBLAS library would be beneficial for this application.
