import math
import os
import time

import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jstats

import gp
import kernels

import resampling

import collectors
import smoothing
import utils


class FeynmanKac:
    """Abstract base class for Feynman-Kac models.

    To actually define a Feynman-Kac model, one must sub-class FeymanKac,
    and define at least the following methods:

        * `M0(self, N)`: returns a collection of N particles generated from the
          initial distribution M_0.
        * `M(self, t, xp)`: generate a collection of N particles at time t,
           generated from the chosen Markov kernel, and given N ancestors (in
           array xp).
        * `logG(self, t, xp, x)`: log of potential function at time t.

    To implement a SQMC algorithm (quasi-Monte Carlo version of SMC), one must
    define methods:

        * `Gamma0(self, u)`: deterministic function such that, if u~U([0,1]^d),
          then Gamma0(u) has the same distribution as X_0
        * `Gamma(self, xp, u)`: deterministic function that, if U~U([0,1]^d)
          then Gamma(xp, U) has the same distribution as kernel M_t(x_{t-1}, dx_t)
          for x_{t-1}=xp

    Usually, a collection of N particles will be simply a numpy array of
    shape (N,) or (N,d). However, this is not a strict requirement, see
    e.g. module `smc_samplers` and the corresponding tutorial in the on-line
    documentation.
    """

    def __init__(
        self,
    ):
        pass

    def _error_msg(self, meth):
        cls_name = self.__class__.__name__
        return f"method/property {meth} missing in class {cls_name}"

    def M0(self, size, key):
        """Sample N times from initial distribution M_0 of the FK model"""
        raise NotImplementedError(self._error_msg("M0"))

    def M(self, t, xp):
        """Generate X_t according to kernel M_t, conditional on X_{t-1}=xp"""
        raise NotImplementedError(self._error_msg("M"))

    def logG(self, t, xp, x, key):
        """Evaluates log of function G_t(x_{t-1}, x_t)"""
        raise NotImplementedError(self._error_msg("logG"))

    def Gamma0(self, u):
        """Deterministic function that transform a uniform variate of dimension
        d_x into a random variable with the same distribution as M0."""
        raise NotImplementedError(self._error_msg("Gamma0"))

    def Gamma(self, t, xp, u):
        """Deterministic function that transform a uniform variate of dimension
        d_x into a random variable with the same distribution as M(self, t, xp).
        """
        raise NotImplementedError(self._error_msg("Gamma"))

    def logpt(self, t, xp, x):
        """Log-density of X_t given X_{t-1}."""
        raise NotImplementedError(err_msg_missing_trans % self.__class__.__name__)

    def default_moments(self, W, X):
        """Default moments (see module ``collectors``).

        Computes weighted mean and variance (assume X is a Numpy array).
        """
        return utils.wmean_and_var(W, X)


class SMC:
    """Metaclass for SMC algorithms.

       Parameters
       ----------
       fk : FeynmanKac object
           Feynman-Kac model which defines which distributions are
           approximated
       T : int, max timoe for simulation
       N : int, optional (default=100)
           number of particles
       qmc : bool, optional (default=False)
           if True use the Sequential quasi-Monte Carlo version (the two
           options resampling and ESSrmin are then ignored)
       resampling : {'multinomial', 'residual', 'stratified', 'systematic', 'ssp'}
           the resampling scheme to be used (see `resampling` module for more
           information; default is 'systematic')
       ESSrmin : float in interval [0, 1], optional
           resampling is triggered whenever ESS / N < ESSrmin (default=0.5)
       store_history : bool, int or callable (default=False)
           whether and when history should be saved; see module `smoothing`
       verbose : bool, optional
           whether to print basic info at every iteration (default=False)
       collect : list of collectors, or 'off' (for turning off summary collections)
           see module ``collectors``

    Attributes
    ----------

       t : int
          current time step
       X : typically a (N,) or (N, d) ndarray (but see documentation)
           the N particles
       A : (N,) ndarray (int)
          ancestor indices: A[n] = m means ancestor of X[n] has index m
       wgts : `Weights` object
           An object with attributes lw (log-weights), W (normalised weights)
           and ESS (the ESS of this set of weights) that represents
           the main (inferential) weights
       aux : `Weights` object
           the auxiliary weights (for an auxiliary PF, see FeynmanKac)
       cpu_time : float
           CPU time of complete run (in seconds)
       hist : `ParticleHistory` object (None if option history is set to False)
           complete history of the particle system; see module `smoothing`
       summaries : `Summaries` object (None if option summaries is set to False)
           each summary is a list of estimates recorded at each iteration. The
           summaries computed by default are ESSs, rs_flags, logLts.
           Extra summaries may also be computed (such as moments and online
           smoothing estimates), see module `collectors`.

       Methods
       -------
       run()
           run the algorithm until completion
       step()
           run the algorithm for one step (object self is an iterator)
    """

    def __init__(
        self,
        T,
        tau,
        fk=None,
        N=100,
        resampler="systematic",
        ESSrmin=0.5,
        store_history=False,
        verbose=False,
        collect=None,
        key=None,
        Y_STEP=1,
    ):

        self.fk = fk
        self.N = N
        self.T = T
        self.tau = tau
        self.ITER_MAX = math.ceil(self.T / self.tau)
        self.ESSrmin = ESSrmin
        self.verbose = False

        if key is None:
            key = jr.key(42)
        rs_key, hist_key = jr.split(key, 2)
        rs = resampling.Resampler(N)

        if resampler == "stratified":
            self.resampling = rs.stratified
        elif resampler == "multinomial":
            self.resampling = rs.multinomial
        elif resampler == "systematic":
            self.resampling = rs.systematic
        elif resampler == "residual":
            self.resampling = rs.residual
        elif resampler == "killing":
            self.resampling = rs.killing
        elif resampler == "ssp":
            self.resampling = rs.ssp
        elif resampler == "parallel":
            if os.uname()[0] != "Linux":
                print(
                    """Parallel resampling only available on Linux,
                    falling back to serial."""
                )
                self.resampling = rs.systematic
            else:
                self.resampling = rs.parallel_systematic
        else:
            self.resampling = rs.systematic
            print(
                """Resampling method must be one of: stratified, multinomial, systematic,
                residual, killing, or ssp, defaulting to systematic """
            )

        # initialisation
        self.t = 0
        self.Y_STEP = Y_STEP
        self.rs_flag = False  # no resampling at time 0, by construction
        self.logLt = 0.0
        self.wgts = resampling.Weights(N=N)
        self.X, self.X_hat, self.A = None, None, None
        self.log_mean_w = None
        self.log_mean_w_prev = None

        # summaries computed at every t
        if collect == "off":
            self.summaries = None
        else:
            self.summaries = collectors.Summaries(collect)
        self.hist = smoothing.generate_hist_obj(store_history, self, hist_key)

    def print_summary(self, ctr):
        print(
            f"i = {ctr:d}: resample: {self.rs_flag}, ESS (end of iter) = {self.wgts.ESS:.2f}"
        )

    @property
    def W(self):
        return self.wgts.lw

    def reset_weights(self):
        self.wgts = resampling.Weights(
            self.N,
        )  # lw=self.wgts.lw)

    def done(self):
        """Time to stop the algorithm"""
        return self.t >= self.T

    def time_to_resample(self):
        """When to resample."""
        return self.wgts.get_ESS() < self.N * self.ESSrmin

    def generate_particles(self, key):
        self.X = self.fk.M0(self.N, key)

    def reweight_particles(self, key, ctr=0):
        if ctr % self.Y_STEP == 0:
            y_idx = int(ctr / self.Y_STEP)
            wts = self.fk.logG(self.t, self.fk.data[y_idx], self.X, key)
        else:
            time_var = (ctr % self.Y_STEP) * self.tau
            # next observation
            y_idx = int((ctr + self.Y_STEP) / self.Y_STEP)
            wts = jstats.norm.logpdf(
                x=self.fk.data[y_idx, 1],
                loc=self.X[:, 1],
                scale=(self.fk.ssm.sigma2_x * time_var),
            )
        self.wgts.add(wts)

    def resample(self, key):
        self.rs_flag = self.time_to_resample()
        if self.rs_flag:  # if resampling
            self.A = self.resampling(self.wgts.W, key)
            self.X_hat = self.X[self.A]
            self.reset_weights()
        else:
            self.A = jnp.arange(self.N)
            self.X_hat = self.X

    def mutate(self, key, ctr=0):
        mvn_key, m_key = jr.split(key, 2)
        ys = self.fk.M(self.t, self.hist.X[-1], m_key, add_noise=False)
        kernel = kernels.RBF(
            length_scale=jnp.array(
                [
                    self.fk.ssm.sigma2_x,
                    self.fk.ssm.sigma2_x,
                    self.fk.ssm.sigma2_x,
                    0.5,
                    0.5,
                ]
            ),
            length_scale_bounds=(1e-4, 1.0),
        )
        gauss_process = gp.GaussianProcessRegressor(kernel=kernel, alpha=1e-2)

        gauss_process.fit(self.hist.X[-1], ys)
        mu = gauss_process.predict(self.X_hat, return_cov=False)
        self.X = mu + 0.001 * jr.normal(mvn_key, shape=self.hist.X[-1].shape)

        tmp = jnp.where(self.X[:, :3] < 1, self.X[:, :3], 1.0 / self.X[:, :3])
        self.X = self.X.at[:, :3].set(tmp)
        tmp = jnp.where(self.X[:, :3] > 0, self.X[:, :3], jnp.abs(self.X[:, :3]) / 10)
        self.X = self.X.at[:, :3].set(tmp)

    def compute_summaries(self, ctr=0):
        if ctr % self.Y_STEP == 0:
            if ctr > 0:
                prec_log_mean_w = self.log_mean_w
            self.log_mean_w = self.wgts.get_log_mean()
            if ctr == 0 or self.rs_flag:
                self.loglt = self.log_mean_w
            else:
                self.loglt = self.log_mean_w - self.log_mean_w_prev
            self.log_mean_w_prev = self.log_mean_w
            self.logLt += self.loglt
        if self.verbose:
            self.print_summary(ctr)
        if self.hist:
            self.hist.save(self)
        # must collect summaries *after* history, because a collector (e.g.
        # FixedLagSmoother) may need to access history
        if self.summaries:
            self.summaries.collect(self)

    def run(self, key, abm=False):
        """Runs particle filter until completion."""
        # initalize eveything at iter = 0
        new_key, obs_key, key = jr.split(key, 3)
        if not abm:  # running regular smc
            self.generate_particles(new_key)
        self.reweight_particles(obs_key)
        self.compute_summaries()

        for ctr in range(1, self.ITER_MAX):
            start = time.time()
            new_key, rs_key, obs_key, key = jr.split(key, 4)
            self.resample(rs_key)
            self.mutate(key, ctr=ctr)
            self.reweight_particles(obs_key, ctr=ctr)
            self.compute_summaries(ctr)
            key = new_key
            end = time.time()
            print(f"Step: {ctr:4d}, Computation time: {end - start:.6f}")
