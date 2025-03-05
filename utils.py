import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd



GPR_CHOLESKY_LOWER = True


def weighted_percentile(data, weights, perc):
    """
    perc : percentile in [0-1]
    """
    ix = jnp.argsort(data)
    data = data[ix]  # sort data
    weights = weights[ix]  # sort weights
    cdf = (jnp.cumsum(weights) - 0.5 * weights) / jnp.sum(
        weights
    )  # 'like' a CDF function
    return jnp.interp(perc, cdf, data)


def proj2(y):
    aux = y.sort()
    n = aux.shape[0]
    tau = aux[0] - 1.0
    i = 1
    while i < n and aux[i] > tau:
        tau += (aux[i] - tau) / (i + 1)
        i += 1
    x = y - tau
    return jnp.where(x > 0, x, 0.0)


def simplex_proj(y, size):
    """Quick projection onto the unit simplex>

    See https://hal.science/hal-01056171v2/document
    and https://arxiv.org/pdf/1101.6081
    """
    u = y.sort(descending=True)
    aux = (jnp.cumsum(u) - 1) / jnp.arange(1, y.shape[0] + 1)
    K = jnp.nonzero(aux < u, size=size)[0][-1]
    x = y - aux[K]
    return jnp.where(x > 0, x, 0.0)


@jax.jit
def pdist(X, sqrt=False):
    dists = jnp.sum((X[:, None] - X[None, :]) ** 2, -1)
    if sqrt:
        return jnp.sqrt(dists)
    return dists


@jax.jit
def cdist(x, y, sqrt=False):
    dists = jnp.sum((x[:, None] - y[None, :]) ** 2, -1)
    if sqrt:
        return jnp.sqrt(dists)
    return dists


def timer(method):
    @functools.wraps(method)
    def timed_method(self, **kwargs):
        starting_time = time.perf_counter()
        out = method(self, **kwargs)
        self.cpu_time = time.perf_counter() - starting_time
        return out

    return timed_method


def load_abm_data(in_file):
    data = pd.read_csv(in_file)
    idxs = np.where(data["hour"] == 0.0)
    data = data.iloc[idxs]
    data = data[["S_susceptible", "I_total_infect", "R_total_recov"]]
    return data.to_numpy() / 4000


def closest_SPD(A):
    """Find the nearest positive-definite matrix to input

    N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    Y = 0.5 * (A + A.T)
    D, Q = np.linalg.eig(Y)
    D[D < 0] = 0

    return Q.dot(np.diag(D)).dot(Q.T)


def is_pos_def(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


def exp_and_normalize(lw):
    w = jnp.exp(lw - lw.max())
    return w / w.sum()


def wmean_and_var(W, x):
    """Component-wise weighted mean and variance.

    Parameters
    ----------
    W : (N,) ndarray
        normalised weights (must be >=0 and sum to one).
    x : ndarray (such that shape[0]==N)
        data

    Returns
    -------
    dictionary
        {'mean':weighted_means, 'var':weighted_variances}
    """

    # m = jnp.average(x, weights=W, axis=0)
    # m2 = jnp.average(x**2, weights=W, axis=0)
    # v = m2 - m**2
    # numerical issues with the computation of v 
    m = W.dot(x)
    tmp = (x - m) ** 2
    v = W.dot(tmp)
    return {"mean": m, "var": v}


def wmean_and_cov(W, x):
    """Weighted mean and covariance matrix.

    Parameters
    ----------
    W : (N,) ndarray
        normalised weights (must be >=0 and sum to one).
    x : ndarray (such that shape[0]==N)
        data

    Returns
    -------
    tuple
        (mean, cov)
    """
    m = np.average(x, weights=W, axis=0)
    cov = np.cov(x.T, aweights=W, ddof=0)

    return m, cov
