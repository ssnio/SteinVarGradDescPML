
#######################################################################
#
# This is a python implementation of Stein Variational Gradient Descent
# heavily copied from Dilin Wang's SVGD repo:
# https://github.com/dilinwang820/Stein-Variational-Gradient-Descent/tree/master/python
#
#######################################################################

# imports
import numpy as np
from scipy.spatial.distance import pdist, squareform


def sq_exp_kernel(theta, h=-1):
    # compute the rbf (squared exponential) kernel
    sq_dist = pdist(theta)
    pairwise_dists = squareform(sq_dist) ** 2
    if h < 0:  # if h < 0, using median trick
        h = np.median(pairwise_dists)
        h = np.sqrt(0.5 * h / np.log(theta.shape[0] + 1))
    kxy = np.exp(-pairwise_dists / h ** 2 / 2)

    # compute the gradient of rbf kernel
    dxkxy = -np.matmul(kxy, theta)
    sumkxy = np.sum(kxy, axis=1)
    for i in range(theta.shape[1]):
        dxkxy[:, i] = dxkxy[:, i] + np.multiply(theta[:, i], sumkxy)
    dxkxy = dxkxy / (h ** 2)
    return kxy, dxkxy


def update(theta, dlogpdf, n_iter=20000, stepsize=1e-2, alpha=0.9):

    num_particles, num_dims = theta.shape

    # adagrad with momentum
    fudge_factor = 1e-6
    historical_grad = np.zeros_like(theta)

    for i in range(n_iter):

        dlogpdf_val = dlogpdf(theta)

        # calculating the kernel matrix
        kxy, dxkxy = sq_exp_kernel(theta)

        # gradient (step direction)
        grad_theta = (np.matmul(kxy, dlogpdf_val) + dxkxy) / num_particles

        # adagrad
        if i == 0:
            historical_grad = grad_theta ** 2
        else:
            historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
        adj_grad = np.divide(grad_theta, fudge_factor + np.sqrt(historical_grad))
        theta = theta + stepsize * adj_grad

    return theta


def update_record(theta, dlogpdf, n_iter=20000, stepsize=1e-2, alpha=0.9):

    num_particles, num_dims = theta.shape
    rec_theta = [theta]
    rec_grad = []

    # adagrad with momentum
    fudge_factor = 1e-6
    historical_grad = np.zeros_like(theta)
    for i in range(n_iter):

        dlogpdf_val = dlogpdf(theta)

        # calculating the kernel matrix
        kxy, dxkxy = sq_exp_kernel(theta)

        kxy_dlogpdf_val = np.matmul(kxy, dlogpdf_val)

        # gradient (step direction)
        grad_theta = (kxy_dlogpdf_val + dxkxy) / num_particles

        # adagrad
        if i == 0:
            historical_grad = grad_theta ** 2
        else:
            historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
        adj_grad = np.divide(grad_theta, fudge_factor + np.sqrt(historical_grad))
        theta = theta + stepsize * adj_grad

        rec_theta.append(theta)
        rec_grad.append([dxkxy, kxy_dlogpdf_val, grad_theta])

    return theta, rec_theta, rec_grad


def d_log_pdf_mvn(mu, cov_mat, theta):
    return - np.matmul(theta - mu, np.linalg.inv(cov_mat))
