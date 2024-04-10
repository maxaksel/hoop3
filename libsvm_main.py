import numpy as np
import matplotlib.pyplot as plt
import time
from funcs.logistic import LogisticFunctions
from util.benchmark import classification_accuracy, libsvm_to_numpy
from opt.opt1 import *
from opt.opt2 import *
from opt.opt3 import *

if __name__ == '__main__':
    dataset = 'mushrooms'

    dimensions = {
        'a9a': 123,
        'australian': 14,
        'australian_scale': 14,
        'ijcnn1.tr': 22,
        'mushrooms': 112,
        'phishing': 68
    }

    X, y = libsvm_to_numpy('./data/' + dataset, dimensions[dataset])
    d = dimensions[dataset]  # dimension of data (not including bias feature)
    N = X.shape[0]

    logistic_problem = LogisticFunctions(data_matrix=X, labels=y, use_gpu=False)

    theta_hat = np.random.rand(d + 1).astype(np.float32)
    orig_theta_hat = np.copy(theta_hat)
    num_iters = 500

    print(type(X))
    print(type(theta_hat))

    # Gradient Descent
    print("Gradient Descent\n=================")
    start_time = time.time()
    loss_o1_q, loss_o1_q_grad, theta_hat = gradient_descent(theta_hat, num_iters, logistic_problem)
    end_time = time.time()
    print("Run time:")
    print(end_time - start_time)
    print(loss_o1_q[-1])

    # plt.semilogy(loss_o1_q.cpu())
    # plt.xlabel(r"Iterations")
    # plt.ylabel(r"$l(\theta; X, y)$")
    # plt.show()

    print("Classification accuracy:")
    print(classification_accuracy(y, X, theta_hat))

    # Accelerated Gradient Descent
    print("Acc. Gradient Descent\n=================")

    theta_hat = np.copy(orig_theta_hat)
    start_time = time.time()
    loss_o1_aq, loss_o1_aq_grad, theta_hat = acc_gradient_descent(theta_hat, num_iters, logistic_problem)
    end_time = time.time()
    print("Run time:")
    print(end_time - start_time)
    print(loss_o1_aq[-1])

    # plt.semilogy(loss_o1_aq.cpu())
    # plt.xlabel(r"Iterations")
    # plt.ylabel(r"$l(\theta; X, y)$")
    # plt.show()

    print("Classification accuracy:")
    print(classification_accuracy(y, X, theta_hat))

    # Quadratic Newton
    print("Quadratic Newton\n=================")

    theta_hat = np.copy(orig_theta_hat)
    start_time = time.time()
    loss_o2_q, loss_o2_q_grad, theta_hat = newton_plain(theta_hat, num_iters, logistic_problem, alpha=10, gamma=1)
    end_time = time.time()
    print("Run time:")
    print(end_time - start_time)
    print(loss_o2_q[-1])

    # plt.semilogy(loss_o2_q.cpu())
    # plt.xlabel(r"Iterations")
    # plt.ylabel(r"$l(\theta; X, y)$")
    # plt.show()

    print("Classification accuracy:")
    print(classification_accuracy(y, X, theta_hat))

    # Slow Cubic Newton
    print("Slow Cubic Newton\n=================")

    theta_hat = np.copy(orig_theta_hat)
    start_time = time.time()
    loss_o2_slow, loss_o2_slow_grad, theta_hat = cubic_newton_slow(theta_hat, num_iters, logistic_problem)
    end_time = time.time()
    print("Run time:")
    print(end_time - start_time)
    print(loss_o2_slow[-1])

    print("Classification accuracy:")
    print(classification_accuracy(y, X, theta_hat))

    # Fast Cubic Newton
    print("Fast Cubic Newton\n=================")

    theta_hat = np.copy(orig_theta_hat)
    start_time = time.time()
    loss_o2_fast, loss_o2_grad_fast, theta_hat = cubic_newton(theta_hat, num_iters, logistic_problem)
    end_time = time.time()
    print("Run time:")
    print(end_time - start_time)

    print(loss_o2_fast[-1])

    # plt.semilogy(loss_o1_q.cpu())
    # plt.xlabel(r"Iterations")
    # plt.ylabel(r"$l(\theta; X, y)$")
    # plt.show()

    print("Classification accuracy:")
    print(classification_accuracy(y, X, theta_hat))

    # Accelerated Cubic Newton
    print("Accelerated Cubic Newton\n=================")

    theta_hat = np.copy(orig_theta_hat)
    start_time = time.time()
    loss_o2_accel, loss_grad_o2_accel, theta_hat = acc_cubic_newton(theta_hat, num_iters, logistic_problem)
    end_time = time.time()
    print("Run time:")
    print(end_time - start_time)

    print(loss_o2_accel[-1])

    # plt.semilogy(loss_o1_q.cpu())
    # plt.xlabel(r"Iterations")
    # plt.ylabel(r"$l(\theta; X, y)$")
    # plt.show()

    print("Classification accuracy:")
    print(classification_accuracy(y, X, theta_hat))

    # Combined Graph
    plt.semilogy(loss_o1_q, label="Gradient Descent")
    plt.semilogy(loss_o1_aq, label="Acc. Gradient Descent")
    plt.semilogy(loss_o2_q, label="Quadratic Newton")
    plt.semilogy(loss_o2_slow, label="Slow Cubic Newton")
    plt.semilogy(loss_o2_fast, label="Fast Cubic Newton")
    plt.semilogy(loss_o2_accel, label="Acc. Cubic Newton")

    plt.xlabel(r"Iterations")
    plt.ylabel(r"$l(\theta; X, y)$")
    plt.legend()
    plt.show()

    plt.figure()
    plt.semilogy(loss_o1_q_grad, label="Gradient Descent")
    plt.semilogy(loss_o1_aq_grad, label="Acc. Gradient Descent")
    plt.semilogy(loss_o2_q_grad, label="Quadratic Newton")
    plt.semilogy(loss_o2_slow_grad, label="Slow Cubic Newton")
    plt.semilogy(loss_o2_grad_fast, label="Fast Cubic Newton")
    plt.semilogy(loss_grad_o2_accel, label="Acc. Cubic Newton")
    plt.xlabel(r"Iterations")
    plt.ylabel(r"$\| \nabla l(\theta; X, y) \|$")
    plt.legend()
    plt.show()
