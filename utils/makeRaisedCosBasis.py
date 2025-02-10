import numpy as np
from scipy.stats import zscore
import pandas as pd

# Generate data


def raisedCosFun(x, ctr, dCtr):
    """
    Raised cosine function

    args:
        x: input
        ctr: center
        dCtr: width (allows assymetry)

    returns:
        y: output
    """
    if not isinstance(dCtr, list):
        dCtr = np.array([dCtr, dCtr])
    left_x = x[x < ctr]
    right_x = x[x >= ctr]
    y_left = 0.5 * (1 + np.cos(np.pi * (left_x - ctr) / dCtr[0]))
    y_right = 0.5 * (1 + np.cos(np.pi * (right_x - ctr) / dCtr[1]))
    y = np.concatenate([y_left, y_right])
    y[x < (ctr - dCtr[0])] = 0
    y[x > (ctr + dCtr[1])] = 0
    return y


def gen_spread(n, n_basis, spread='linear', **kwargs):
    """
    Generate spread for basis functions

    args:
        n: number of time bins
        n_basis: number of basis functions
        spread: 'linear' or 'log'
        kwargs: additional arguments
            - a: sigmoid scale
            - b: sigmoid shift

    returns:
        ctrs: centers of basis functions
        dctrs: widths of basis functions
    """
    if spread == 'linear':
        ctrs = np.linspace(0, n-2, n_basis)[1:-1]
        ctrs = np.concatenate([[0], ctrs, [n]])
        dctrs = np.diff(ctrs)[0]
        dctrs = [[dctrs, dctrs]] * (n_basis)

    if spread == 'log':
        ctrs = np.logspace(0, np.log10(n-1), n_basis)[1:-1]
        ctrs = np.concatenate([[0], ctrs, [n]])
        dctrs = np.diff(ctrs)
        # make assymetric functions such that touching sides have same width
        dctrs = np.concatenate([[0.1], dctrs, [0.1]])
        dctrs = np.stack([dctrs[:-1], dctrs[1:]]).T

    if spread == 'sigmoid':
        # Check if a and b are provided
        if 'a' in kwargs.keys():
            a = kwargs['a']
        else:
            raise ValueError('Sigmoid scale (a) must be provided')
        if 'b' in kwargs.keys():
            b = kwargs['b']
        else:
            raise ValueError('Sigmoid shift (b) must be provided')
        raw_ctrs = np.linspace(0, n, n_basis)
        # sigmoid = lambda x, a: (2 / (1 + np.exp(-a*x))) - 1
        def sigmoid(x, a, b): return 1 / (1 + np.exp(-a*(x-b)))
        # ctrs_deltas = sigmoid(raw_ctrs, 0.005, 1000)
        ctrs_deltas = sigmoid(raw_ctrs, a, b)
        ctrs = np.cumsum(ctrs_deltas)
        # Rescale ctrs to be between 0 and n
        ctrs = (ctrs - ctrs.min()) / (ctrs.max() - ctrs.min()) * n
        # Calculate widths
        dctrs = np.diff(ctrs)
        dctrs = np.concatenate([[0.1], dctrs, [0.1]])
        dctrs = np.stack([dctrs[:-1], dctrs[1:]]).T

        # fig, ax = plt.subplots(2,1)
        # ax[0].plot(ctrs_deltas, 'o')
        # ax[1].plot(ctrs, 'o')
        # plt.show()

    return ctrs, dctrs

# linear_ctrs, linear_dctrs = gen_spread(1000, 10, spread='linear')
# log_ctrs, log_dctrs = gen_spread(1000, 10, spread='log')
#
# plt.plot(linear_ctrs, np.zeros_like(linear_ctrs), 'o', label='Linear')
# plt.scatter(log_ctrs, np.ones_like(log_ctrs), label='Log', alpha=0.5, linewidth=1, edgecolor='k')
# plt.show()


def gen_raised_cosine_basis(n, ctrs, dctrs):
    """
    Generate raised cosine basis functions

    args:
        ctrs: centers of basis functions
        dctrs: widths of basis functions

    returns:
        basis: n_basis x n matrix of basis functions
    """
    basis_funcs = np.stack(
        [
            raisedCosFun(
                np.arange(n),
                this_ctr,
                list(this_dctrs)) for this_ctr, this_dctrs in zip(ctrs, dctrs)])

    return basis_funcs


if __name__ == '__main__':
    import pylab as plt
    n = 1000
    n_basis = 10
    lin_ctrs, lin_dctrs = gen_spread(n, n_basis, spread='linear')
    log_ctrs, log_dctrs = gen_spread(n, n_basis, spread='log')
    sig_ctrs, sig_dctrs = gen_spread(
        n, n_basis, spread='sigmoid', a=0.02, b=200)
    linear_basis_funcs = gen_raised_cosine_basis(n, lin_ctrs, lin_dctrs)
    log_basis_funcs = gen_raised_cosine_basis(n, log_ctrs, log_dctrs)
    sigmoid_basis_funcs = gen_raised_cosine_basis(n, sig_ctrs, sig_dctrs)

    fig, ax = plt.subplots(3, 2, sharex='col', sharey=True, figsize=(10, 10))
    ax[0, 0].plot(linear_basis_funcs.T, color='k')
    ax[0, 0].plot(linear_basis_funcs.sum(axis=0), color='r', linewidth=3)
    ax[0, 0].set_title('Linearly spaced basis functions')
    ax[1, 0].plot(log_basis_funcs.T, color='k')
    ax[1, 0].plot(log_basis_funcs.sum(axis=0), color='r', linewidth=3)
    ax[1, 0].set_title('Logarithmically spaced basis functions')
    ax[0, 1].plot(linear_basis_funcs.T, color='k')
    ax[0, 1].plot(linear_basis_funcs.sum(axis=0), color='r', linewidth=3)
    ax[0, 1].set_title('Linearly spaced basis functions')
    ax[1, 1].plot(log_basis_funcs.T, color='k')
    ax[1, 1].plot(log_basis_funcs.sum(axis=0), color='r', linewidth=3)
    ax[1, 1].set_title('Logarithmically spaced basis functions')
    ax[2, 0].plot(sigmoid_basis_funcs.T, color='k')
    ax[2, 0].plot(sigmoid_basis_funcs.sum(axis=0), color='r', linewidth=3)
    ax[2, 0].set_title('Sigmoidally spaced basis functions')
    ax[2, 1].plot(sigmoid_basis_funcs.T, color='k')
    ax[2, 1].plot(sigmoid_basis_funcs.sum(axis=0), color='r', linewidth=3)
    ax[2, 1].set_title('Sigmoidally spaced basis functions')
    ax[0, 1].set_xscale('log')
    ax[1, 1].set_xscale('log')
    ax[-1, 0].set_xlabel('Time (linear scale)')
    ax[-1, 1].set_xlabel('Time (log scale)')
    ax[0, 0].set_xlim([0, n])
    ax[0, 1].set_xlim([0, n])
    plt.tight_layout()
    plt.show()
