# pylint: disable=C0103
import ctypes as ct
import functools
import os
import random

import jax
import jax.numpy as jnp
import jax.random as jr

import numpy as np
import numpy.ctypeslib as npct

if os.uname()[0] == "Linux":
    # setup interface with the scan c-lib
    _float_ptr = npct.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
    _scan = npct.load_library("parallel-scan/libthrustscan.so", os.path.dirname(__file__))
    # Define the return type of the C function
    _scan.scan.restype = ct.c_float
    # Define arguments of the C function
    _scan.scan.argtypes = [_float_ptr, _float_ptr, ct.c_int, ct.c_bool]



def offspring_to_ancestor(O, A, N):
    for i in range(1, N):
        if i == 1:
            start = 0
        else:
            start = O[i - 1]
        o = (O[i] - start).astype(int)
        for j in range(1, o+1):
            A = A.at[start+j].set(i)
    return A


# @numba.jit(nopython=True)
def inverse_cdf(su, W, M):
    """Inverse CDF algorithm for a finite distribution.

    Parameters
    ----------
    su  : (M,) ndarray
        M sorted uniform variates (i.e. M ordered points in [0,1]).
    W : (N,) ndarray
        a vector of N normalized weights (>=0 and sum to one)

    Returns
    -------
    A : (M,) ndarray
        a vector of M indices in range 0, ..., N-1
    """
    j = 0
    s = W[0]
    # M = su.shape[0]
    A = np.empty(M, dtype=np.int32)
    for n in range(M):
        while su[n] > s:
            j += 1
            s += W[j]
        A[n] = j
        print(s)
    return A


def multinomial(W, M, key):
    """Multinomial resampling.

    Popular resampling scheme, which amounts to sample N independently from
    the multinomial distribution that generates n with probability W^n.

    This resampling scheme is *not* recommended for various reasons; basically
    schemes like stratified / systematic / SSP tends to introduce less noise,
    and may be faster too (in particular systematic).

    Note
    ----
    As explained in the book, the output of this function is ordered. This is
    fine in most practical cases, but, in case you need truly IID samples, use
    `multinomial_iid` instead (which calls `multinomial`, and randomly
    permutate the result).
    """
    return inverse_cdf(uniform_spacings(M, key), W, M)


def uniform_spacings(N, key):
    """Generate ordered uniform variates in O(N) time.

    Parameters
    ----------
    N : int (>0)
        the expected number of uniform variates

    Returns
    -------
    (N,) float ndarray
        the N ordered variates (ascending order)

    Note
    ----
    This is equivalent to::

        from numpy import random
        u = sort(random.rand(N))

    but the line above has complexity O(N*log(N)), whereas the algorithm
    used here has complexity O(N).

    """
    z = jnp.cumsum(-jnp.log(jr.uniform(key, shape=(N + 1,))))
    return z[:-1] / z[-1]


def multinomial_once(W, key):
    """Sample once from a Multinomial distribution.

    Parameters
    ----------
    W : (N,) ndarray
        normalized weights (>=0, sum to one)

    Returns
    -------
    int
        a single draw from the discrete distribution that generates n with
        probability W[n]

    Note
    ----
    This is equivalent to

       A = multinomial(W, M=1)

    but it is faster.
    """
    # return np.searchsorted(np.cumsum(W), random.rand())
    return jnp.searchsorted(jnp.cumsum(W), jr.uniform(key))


def multinomial_iid(W, M=None):
    """Multinomial resampling (IID draws).

    Same as multinomial resampling, except the output is randomly permuted, to
    ensure that the resampled indices are IID (independent and identically
    distributed).

    """
    A = multinomial(W, M=M)
    random.shuffle(A)
    return A


class Weights:
    """A class to store N log-weights, and automatically compute normalised
    weights and their ESS.

    Parameters
    ----------
    lw : (N,) array or None
        log-weights (if None, object represents a set of equal weights)

    Attributes
    ----------
    lw : (N), array
        log-weights (un-normalised)
    W : (N,) array
        normalised weights
    ESS : scalar
        the ESS (effective sample size) of the weights

    Warning
    -------
    Objects of this class should be considered as immutable; in particular,
    method add returns a *new* object. Trying to modifying directly the
    log-weights may introduce bugs.

    """

    def __init__(self, N, lw=None):
        self.N = N
        self.log_mean = None
        if lw is None:
            self.lw = jnp.zeros(N)
        else:
            self.lw = lw
        self.ESS = self.get_ESS()

    def get_log_mean(self):
        m = self.lw.max()
        w = jnp.exp(self.lw - m)
        self.log_mean = m + jnp.log(w.sum() / self.N)
        return self.log_mean

    def get_ESS(self):
        """Computes ESS from log weights.

        As a side effect, stores normalized weights in attribute self.W.
        """
        self.lw = self.lw.at[jnp.isnan(self.lw)].set(-jnp.inf)
        m = self.lw.max()
        w = jnp.exp(self.lw - m)
        s = w.sum()
        self.W = w / s
        self.ESS = 1.0 / jnp.sum(self.W**2)
        return self.ESS

    # @jit
    def add(self, delta):
        """Increment weights: lw <-lw + delta.

        Parameters
        ----------
        delta : (N,) array
            incremental log-weights

        """
        for i in range(self.N):
            self.lw = self.lw.at[i].add(delta[i])
        # return self.lw

        # if self.lw is None:
        #     return self.__class__(lw=delta)
        # return self.__class__(lw=self.lw + delta)


class Resampler:
    def __init__(self, N):
        self.N = N  # num particles
        self.M = N
        self.su = jnp.empty(self.M)
        self.A = jnp.empty(self.M, dtype=jnp.int32)

    def inverse_cdf(self, su, W):
        """Inverse CDF algorithm for a finite distribution.

        Parameters
        ----------
        su  : (M,) ndarray
            M sorted uniform variates (i.e. M ordered points in [0,1]).
        W : (N,) ndarray
            a vector of N normalized weights (>=0 and sum to one)

        Returns
        -------
        A : (M,) ndarray
            a vector of M indices in range 0, ..., N-1
        """
        self.A = self.A.at[jnp.arange(self.M)].set(0)
        j = 0
        s = W[0]
        M = su.shape[0]
        for n in range(M):
            while su[n] > s:
                j += 1
                s += W[j]
            self.A = self.A.at[n].set(j)
        return self.A

    def uniform_spacings(self, key, N):
        """Generate ordered uniform variates in O(N) time.

        Parameters
        ----------
        N : int (>0)
            the expected number of uniform variates

        Returns
        -------
        (N,) float ndarray
            the N ordered variates (ascending order)

        Note
        ----
        This is equivalent to::

            from numpy import random
            u = sort(random.rand(N))

        but the line above has complexity O(N*log(N)), whereas the algorithm
        used here has complexity O(N).

        """
        z = jnp.cumsum(-jnp.log(jr.uniform(key, shape=(N + 1,))))
        return z[:-1] / z[-1]

    def multinomial(self, W, M, key):
        """Multinomial resampling.

        Popular resampling scheme, which amounts to sample N independently from
        the multinomial distribution that generates n with probability W^n.

        This resampling scheme is *not* recommended for various reasons; basically
        schemes like stratified / systematic / SSP tends to introduce less noise,
        and may be faster too (in particular systematic).

        Note
        ----
        As explained in the book, the output of this function is ordered. This is
        fine in most practical cases, but, in case you need truly IID samples, use
        `multinomial_iid` instead (which calls `multinomial`, and randomly
        permutate the result).
        """

        self.su = self.uniform_spacings(self.M, key)
        A = inverse_cdf(self.su, W, self.su.shape[0])
        return A

    def stratified(self, W, M, key):
        """Stratified resampling."""
        self.su = (jr.uniform(key, shape=(M,)) + jnp.arange(M)) / M
        A = inverse_cdf(self.su, W, self.su.shape[0])
        return A

    def parallel_systematic(self, W, key):
        """Systematic resampling method with parallel-prefix sum.

        See Kitagawa, 1996: https://doi-org.utk.idm.oclc.org/10.2307/1390750
        deterministic alg in appendix a-D
        """

        # compute cumulative sum of normalized weights
        N = W.shape[0]
        wts_out = np.zeros(N, dtype=np.float32)
        _ = _scan.scan(
            wts_out, np.asarray(W, dtype=np.float32), ct.c_int(N), ct.c_bool(True)
        )
        R = jnp.floor(jr.uniform(key) + (N * wts_out / wts_out[-1]))
        # cumulative offspring
        O = jnp.where(N > R, R, N).astype(jnp.int32)
        A = jnp.zeros(shape=N, dtype=jnp.int32)
        A = offspring_to_ancestor(O, A, N)

        return A

    def systematic(self, W, key):
        """Systematic resampling."""
        # if W is None:
        #     W = self.W
        # if M is None:
        M = self.N
        # NOTE: do we need to check if W is normalized to 1?
        # print(W.sum())
        # su = (0.0796543 + jnp.arange(M)) / M
        su = (jr.uniform(key) + jnp.arange(M)) / M
        A = self.inverse_cdf(su, W)
        return A

    def residual(self, key, W, M):
        """Residual resampling."""
        N = W.shape[0]
        A = jnp.empty(M, dtype=jnp.int32)
        MW = M * W
        intpart = jnp.floor(MW).astype(jnp.in32)
        sip = jnp.sum(intpart)
        res = MW - intpart
        sres = M - sip
        A[:sip] = jnp.arange(N).repeat(intpart)
        # each particle n is repeated intpart[n] times
        if sres > 0:
            A[sip:] = self.multinomial(key, res / sres, M=sres)
        return A

    @staticmethod
    # @numba.njit
    def ssp(W, M):
        """SSP resampling.

        SSP stands for Srinivasan Sampling Process. This resampling scheme is
        discussed in Gerber et al (2019). Basically, it has similar properties as
        systematic resampling (number of off-springs is either k or k + 1, with
        k <= N W^n < k +1), and in addition is consistent. See that paper for more
        details.

        Reference
        =========
        Gerber M., Chopin N. and Whiteley N. (2019). Negative association, ordering
        and convergence of resampling methods. Ann. Statist. 47 (2019), no. 4, 2236–2260.
        """
        N = W.shape[0]
        MW = M * W
        nr_children = np.floor(MW).astype(np.int32)
        xi = MW - nr_children
        u = np.random.random_sample(size=N - 1)
        i, j = 0, 1
        for k in range(N - 1):
            delta_i = min(xi[j], 1.0 - xi[i])  # increase i, decr j
            delta_j = min(xi[i], 1.0 - xi[j])  # the opposite
            sum_delta = delta_i + delta_j
            # prob we increase xi[i], decrease xi[j]
            pj = delta_i / sum_delta if sum_delta > 0.0 else 0.0
            # sum_delta = 0. => xi[i] = xi[j] = 0.
            if u[k] < pj:  # swap i, j, so that we always inc i
                j, i = i, j
                delta_i = delta_j
            if xi[j] < 1.0 - xi[i]:
                xi[i] += delta_i
                j = k + 2
            else:
                xi[j] -= delta_i
                nr_children[i] += 1
                i = k + 2
        # due to round-off error accumulation, we may be missing one particle
        if np.sum(nr_children) == M - 1:
            last_ij = i if j == k + 2 else j
            if xi[last_ij] > 0.99:
                nr_children[last_ij] += 1
        if np.sum(nr_children) != M:
            # file a bug report with the vector of weights that causes this
            raise ValueError("ssp resampling: wrong size for output")
        return np.arange(N).repeat(nr_children)

    def killing(self, W, M):
        """Killing resampling.

        This resampling scheme was not described in the book. For each particle,
        one either keeps the current value (with probability W[i] / W.max()), or
        replaces it by a draw from the multinomial distribution.

        This scheme requires to take M=N.
        """
        N = W.shape[0]
        if M != N:
            raise ValueError("killing resampling defined only for M=N")
        killed = jr.unifrom(shape=(N,)) * W.max() >= W
        nkilled = killed.sum()
        A = jnp.arange(N)
        A[killed] = self.multinomial(W, nkilled)
        return A
