import functools
import time


import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from jax.scipy.linalg import cho_solve, cholesky, solve_triangular

# from kernels import RBF, Matern, Kernel

GPR_CHOLESKY_LOWER = True


def weighted_percentile(data, weights, perc):
    """
    perc : percentile in [0-1]!
    """
    ix = jnp.argsort(data)
    data = data[ix] # sort data
    weights = weights[ix] # sort weights
    cdf = (jnp.cumsum(weights) - 0.5 * weights) / jnp.sum(weights) # 'like' a CDF function
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
    aux = (jnp.cumsum(u)-1)/ jnp.arange(1, y.shape[0]+1)
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


def gp_interpolate(X_prev, y_prev, X, sigma, len_scale):
    L, alpha = gp_fit(X_prev, y_prev, sigma, len_scale)
    # print(alpha)
    return gp_predict(X, X_prev, L, alpha, len_scale)


def gp_kernel(X, len_scale, Y=None):
    if Y is None:
        dists = pdist(X / len_scale)
    else:
        dists = cdist(X / len_scale, Y / len_scale)
    K = jnp.exp(-0.5 * dists)
    return K


def gp_predict(X_prev, X, L, alpha, len_scale):
    K_t = gp_kernel(X_prev, len_scale, X)
    y_mean = K_t @ alpha
    # Alg 2.1, page 19, line 5 -> v = L \ K(X_test, X_train)^T
    V = solve_triangular(
        L, K_t.T, lower=GPR_CHOLESKY_LOWER, check_finite=False
    )
    y_cov = gp_kernel(X, len_scale) - V.T @ V
    return y_mean, y_cov


def gp_fit(X_train, y_train, sigma, len_scale):
    K = gp_kernel(X_train, len_scale)
    print(K.shape)
    print(XXX)
    K = K.at[jnp.diag_indices_from(K)].add(sigma)
    L = cholesky(K, lower=GPR_CHOLESKY_LOWER)
    if jnp.isnan(L).any():
        print(
                f"The kernelis not returning a positive "
                "definite matrix. Try gradually increasing the 'alpha' "
                "parameter of your GaussianProcessRegressor estimator."
        )
        print(K)
        print(X_train)
        L = closest_SPD(L)
        # Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ y)
    alpha = cho_solve(
        (L, GPR_CHOLESKY_LOWER),
        y_train,
    )
    return (L, alpha)


def timer(method):
    @functools.wraps(method)
    def timed_method(self, **kwargs):
        starting_time = time.perf_counter()
        out = method(self, **kwargs)
        self.cpu_time = time.perf_counter() - starting_time
        return out

    return timed_method


def load_abm_data(in_file):
    # data = np.loadtxt(in_file, delimiter=",", skiprows=1, usecols=(1,2,3,4,6))
    data = pd.read_csv(in_file)
    idxs = np.where(data['hour'] == 0.0)
    data = data.iloc[idxs]
    data = data[['S_susceptible', 'I_total_infect', 'R_total_recov']]
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
    # print(lw.max(), lw.min())
    # w = np.exp(lw - lw.sum())
    w = np.exp(lw - lw.max())
    return w / w.sum()


def log_sum_exp(v):
    """Log of the sum of the exp of the arguments.

    Parameters
    ----------
    v : ndarray

    Returns
    -------
    l : float
        l = log(sum(exp(v)))

    Note
    ----
    use the log_sum_exp trick to avoid overflow: i.e. we remove the max of v
    before exponentiating, then we add it back

    See also
    --------
    log_mean_exp

    """
    m = v.max()
    return m + np.log(np.sum(np.exp(v - m)))


def log_sum_exp_ab(a, b):
    """log_sum_exp for two scalars.

    Parameters
    ----------
    a, b : float

    Returns
    -------
    c : float
        c = log(e^a + e^b)
    """
    if a > b:
        return a + np.log1p(np.exp(b - a))
    else:
        return b + np.log1p(np.exp(a - b))


def log_mean_exp(v, W=None):
    """Returns log of (weighted) mean of exp(v).

    Parameters
    ----------
    v : ndarray
        data, should be such that v.shape[0] = N

    W : (N,) ndarray, optional
        normalised weights (>=0, sum to one)

    Returns
    -------
    ndarray
        mean (or weighted mean, if W is provided) of vector exp(v)

    See also
    --------
    log_sum_exp

    """
    m = v.max()
    V = np.exp(v - m)
    if W is None:
        return m + np.log(np.mean(V))
    else:
        return m + np.log(np.average(V, weights=W))


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
    # numerical issues with the above
    m = W.dot(x)
    tmp = (x - m)**2
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


def wmean_and_var_str_array(W, x):
    """Weighted mean and variance of each component of a structured array.

    Parameters
    ----------
    W : (N,) ndarray
        normalised weights (must be >=0 and sum to one).
    x : (N,) structured array
        data

    Returns
    -------
    dictionary
        {'mean':weighted_means, 'var':weighted_variances}
    """
    m = np.empty(shape=x.shape[1:], dtype=x.dtype)
    v = np.empty_like(m)
    for p in x.dtype.names:
        m[p], v[p] = wmean_and_var(W, x[p]).values()
    return {"mean": m, "var": v}


def _wquantiles(W, x, alphas):
    N = W.shape[0]
    order = np.argsort(x)
    cw = np.cumsum(W[order])
    indices = np.searchsorted(cw, alphas)
    quantiles = []
    for a, n in zip(alphas, indices):
        prev = np.clip(n - 1, 0, N - 2)
        q = np.interp(a, cw[prev : prev + 2], x[order[prev : prev + 2]])
        quantiles.append(q)
    return quantiles


def wquantiles(W, x, alphas=(0.25, 0.50, 0.75)):
    """Quantiles for weighted data.

    Parameters
    ----------
    W : (N,) ndarray
        normalised weights (weights are >=0 and sum to one)
    x : (N,) or (N,d) ndarray
        data
    alphas : list-like of size k (default: (0.25, 0.50, 0.75))
        probabilities (between 0. and 1.)

    Returns
    -------
    a (k,) or (d, k) ndarray containing the alpha-quantiles
    """
    if len(x.shape) == 1:
        return _wquantiles(W, x, alphas=alphas)
    elif len(x.shape) == 2:
        return np.array(
            [_wquantiles(W, x[:, i], alphas=alphas) for i in range(x.shape[1])]
        )


def wquantiles_str_array(W, x, alphas=(0.25, 0.50, 0, 75)):
    """quantiles for weighted data stored in a structured array.

    Parameters
    ----------
    W : (N,) ndarray
        normalised weights (weights are >=0 and sum to one)
    x : (N,) structured array
        data
    alphas : list-like of size k (default: (0.25, 0.50, 0.75))
        probabilities (between 0. and 1.)

    Returns
    -------
    dictionary {p: quantiles} that stores for each field name p
    the corresponding quantiles

    """
    return {p: wquantiles(W, x[p], alphas) for p in x.dtype.names}