# pylint: disable=C0103

import warnings

from functools import partial

from scipy import stats

from sklearn.exceptions import ConvergenceWarning

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jstats

import diffrax

# import rk_solvers
import utils
from fk import FeynmanKac


warnings.filterwarnings("ignore", category=ConvergenceWarning)


class StateSpaceModel:
    """Base class for state-space models.

    To define a state-space model class, you must sub-class `StateSpaceModel`,
    and at least define methods PX0, PX, and PY. Here is an example::

        class LinearGauss(StateSpaceModel):
            def PX0(self):  # The law of X_0
                return dists.Normal(scale=self.sigmaX)
            def PX(self, t, xp):  # The law of X_t conditional on X_{t-1}
                return dists.Normal(loc=self.rho * xp, scale=self.sigmaY)
            def PY(self, t, xp, x):  # the law of Y_t given X_t and X_{t-1}
                return dists.Normal(loc=x, scale=self.sigmaY)

    These methods return ``ProbDist`` objects, which are defined in the module
    `distributions`. The model above is a basic linear Gaussian SSM; it
    depends on parameters rho, sigmaX, sigmaY (which are attributes of the
    class). To define a particular instance of this class, we do::

        a_certain_ssm = LinearGauss(rho=.8, sigmaX=1., sigmaY=.2)

    All the attributes that appear in ``PX0``, ``PX`` and ``PY`` must be
    initialised in this way. Alternatively, it it possible to define default
    values for these parameters, by defining class attribute
    ``default_params`` to be a dictionary as follows::

        class LinearGauss(StateSpaceModel):
            default_params = {'rho': .9, 'sigmaX': 1., 'sigmaY': .1}
            # rest as above

    Optionally, we may also define methods:

    * `proposal0(self, data)`: the (data-dependent) proposal dist at time 0
    * `proposal(self, t, xp, data)`: the (data-dependent) proposal distribution at
      time t, for X_t, conditional on X_{t-1}=xp
    * `logeta(self, t, x, data)`: the auxiliary weight function at time t

    You need these extra methods to run a guided or auxiliary particle filter.

    """

    def __init__(self, **kwargs):
        if hasattr(self, "default_params"):
            self.__dict__.update(self.default_params)
        self.__dict__.update(kwargs)

    def _error_msg(self, method):
        return f"method {method} not implemented in class {self.__class__.__name__}"

    @classmethod
    def state_container(cls, N, T):
        law_X0 = cls().PX0()
        dim = law_X0.dim
        shape = [N, T]
        if dim > 1:
            shape.append(dim)
        return jnp.empty(shape, dtype=law_X0.dtype)

    def PX0(self):
        "Law of X_0 at time 0"
        raise NotImplementedError(self._error_msg("PX0"))

    def PX(self, t, xp, key, add_noise=True):
        "Law of X_t at time t, given X_{t-1} = xp"
        raise NotImplementedError(self._error_msg("PX"))

    def PY(self, t, xp, x, idx, key):
        """Conditional distribution of Y_t, given the states."""
        raise NotImplementedError(self._error_msg("PY"))

    def PY_rvs(self, t, x, key):
        """Conditional distribution of Y_t, given the states."""
        raise NotImplementedError(self._error_msg("PY"))

    # def proposal0(self, data):
    #     raise NotImplementedError(self._error_msg("proposal0"))

    # def proposal(self, t, xp, data):
    #     """Proposal kernel (to be used in a guided or auxiliary filter).

    #     Parameter
    #     ---------
    #     t: int
    #         time
    #     x:
    #         particles
    #     data: list-like
    #         data
    #     """
    #     raise NotImplementedError(self._error_msg("proposal"))

    # def upper_bound_log_pt(self, t):
    #     """Upper bound for log of transition density.

    #     See `smoothing`.
    #     """
    #     raise NotImplementedError(err_msg_missing_cst % self.__class__.__name__)

    # def add_func(self, t, xp, x):
    #     """Additive function."""
    #     raise NotImplementedError(self._error_msg("add_func"))

    def simulate_given_x(self, t, x, key):
        # lag_x = [None] + x[:-1]
        vals = []
        for _t, _x in enumerate(x):
            new_key, key = jr.split(key, 2)
            vals.append(self.PY_rvs(_t, _x, key))
            key = new_key
        return jnp.array(vals)

    def simulate(self, T, key, simulate_obs=False, add_noise=True):
        """Simulate state and observation processes.

        Parameters
        ----------
        T: int
            processes are simulated from time 0 to time T-1

        Returns
        -------
        x, y: lists
            lists of length T
        """
        x = [self.PX0()]
        y = None
        if simulate_obs:
            key, obs_key = jr.split(key, 2)
        for t in range(1, T):
            new_key, key = jr.split(key, 2)
            x.append(self.PX(t, x[-1], key, add_noise))
            key = new_key
        if simulate_obs:
            y = self.simulate_given_x(T, x, obs_key)
            y = jnp.ravel(y)
        return x, y


class Bootstrap(FeynmanKac):
    """Bootstrap Feynman-Kac formalism of a given state-space model.

    Parameters
    ----------

    ssm: `StateSpaceModel` object
        the considered state-space model
    data: list-like
        the data

    Returns
    -------
    `FeynmanKac` object
        the Feynman-Kac representation of the bootstrap filter for the
        considered state-space model
    """

    def __init__(self, ssm=None, data=None):
        self.ssm = ssm
        self.data = data
        self.du = self.ssm.PX0().ndim

    @property
    def T(self):
        return 0 if self.data is None else len(self.data)

    def M0(self, N, key):
        return self.ssm.PX0_rvs(N, key)

    def M(self, t, xp, key, add_noise=True):
        return self.ssm.PX_rvs(t, xp, key, add_noise)

    def logG(self, t, xp, x, key):
        res = self.ssm.PY_logpdf(t, self.data[t], x, key)
        return res

    def Gamma0(self, u):
        return self.ssm.PX0().ppf(u)

    def Gamma(self, t, xp, u):
        return self.ssm.PX(t, xp).ppf(u)

    def logpt(self, t, xp, x):
        """PDF of X_t|X_{t-1}=xp"""
        return self.ssm.PX_logpdf(t, xp, x)

    def upper_bound_trans(self, t):
        return self.ssm.upper_bound_log_pt(t)

    def add_func(self, t, xp, x):
        return self.ssm.add_func(t, xp, x)


class SIRODE(StateSpaceModel):
    def __init__(
        self,
        N,
        I0,
        log_beta,
        log_gamma,
        sigma2_y,
        sigma2_x,
        tau,
        seed,
        abm_file="./data/abm_data/counts_1.csv",
    ):
        super().__init__()
        # population size
        self.N = N
        # ODE ICs
        self.I0 = I0
        self.Y0 = None

        self.log_beta = log_beta
        self.log_gamma = log_gamma
        self.param_dim = 2

        # setup jax random
        self.ic_key = jr.PRNGKey(seed)
        # ode solver
        self.rk = rk_solvers.RK()
        self.tau = tau
        # get abm data
        self.y = utils.load_abm_data(abm_file)
        # model and obs noise
        self.sigma2_x = sigma2_x
        self.sigma2_y = sigma2_y

        # self.n_particles = n_particles

    def PX0(self):
        """Set prior over initial states for a single sample."""
        S = self.N

        u_key, n_key = jr.split(self.ic_key, 2)

        loc = 0.00025
        scale = 0.0005
        lb = 0.0
        ub = 0.001
        lb_trans, ub_trans = (lb - loc) / scale, (ub - loc) / scale
        R = stats.truncnorm(lb_trans, ub_trans, loc=loc, scale=scale).rvs()

        R0 = R * S
        S = S - R0
        I = jnp.log(self.I0) + jr.uniform(u_key, minval=-10.0, maxval=-5.5)
        I0 = jnp.exp(I + jnp.log(S))
        S0 = S - I0
        Y0 = jnp.array([S0, I0, R0])

        self.Y0 = Y0 + self.sigma2_y * jr.normal(n_key, shape=(3,))
        self.Y0 = jnp.where(self.Y0 > 0, self.Y0, jnp.abs(self.Y0))
        self.Y0 = jnp.where(self.Y0 < 1.0, self.Y0, 1.0)

        self.Y0 = self.Y0 / self.Y0.sum()
        return self.Y0

    def PX0_rvs(self, size, key):
        """Set prior over initial states for multiple samples."""
        if self.Y0 is not None:
            Y0 = self.Y0.copy()
        else:
            Y0 = self.PX0()
        Y0 = Y0 + self.sigma2_y * jr.normal(key, shape=(size, 3))
        Y0 = jnp.where(Y0 > 0, Y0, jnp.abs(Y0))
        Y0 = jnp.where(Y0 < 1.0, Y0, 1.0)
        Y0 = jax.vmap(lambda x: x / x.sum(), in_axes=0)(Y0)
        return Y0

    def PY(self, t, xp, x, idx, key):
        # taking the abm output + Gaussian noise
        return self.y[idx, 1] + self.sigma2_y * jr.normal(key)

    def PY_rvs(self, t, x, key):
        return x[1] + self.sigma2_y * jr.normal(key)

    def PY_logpdf(self, t, y, x):
        res = jax.vmap(jstats.norm.logpdf, in_axes=(0, None, None))(
            x[:, 1], y[1], self.sigma2_y
        )

        return res

    def _sir(self, t, y, args=None):
        beta = jnp.exp(self.log_beta)
        # beta = varying_beta(t)
        gamma = jnp.exp(self.log_gamma)
        dS = -(beta * y[0] * y[1])
        dI = (beta * y[0] * y[1]) - y[1] * gamma
        dR = y[1] * gamma
        return jnp.array([dS, dI, dR])

    def PX(self, t, xp, key, add_noise=True):
        """Compute single sample of the push-forward."""
        res = self.rk.step(
            self._sir,
            [0, 1],
            xp,
            1,
        )
        noise = 0.0
        if add_noise:
            noise = jr.normal(key, shape=res.shape) * self.sigma2_x
        return res + noise

    def PX_logpdf(self, t, xp, x, key):
        """Compute logpdf of SIR model states."""
        print("x", x)
        print("xp", xp)
        # res = jax.vmap()

    def PX_rvs(self, t, xp, key):
        """Compute an n_particle sample from the push-forward."""
        res = jax.vmap(self.rk.step, in_axes=(None, None, 0, None))(
            self._sir, [0, self.tau], xp, 1
        )
        res += jr.normal(key, shape=res.shape) * self.sigma2_x
        res = jnp.where(res > 0.0, res, jnp.abs(res))
        res = jnp.where(res < 1.0, res, 1 - jnp.abs(1 - res))
        return res


class SIRSDE(StateSpaceModel):
    def __init__(
        self,
        N,
        I0,
        sigma2_y,
        sigma2_x,
        tau,
        key,
        abm_file=None,  # "./data/abm_data/counts_1.csv",
    ):
        super().__init__()
        # population size
        self.N = N
        # ICs
        self.I0 = I0
        self.Y0 = None
        # jax random
        self.ic_key, self.obs_key, self.X_key, self.theta_key = jr.split(key, 4)
        # ode solver
        # self.rk = rk_solvers.RK()
        self.tau = tau
        # get abm data
        if abm_file is not None:
            self.y = utils.load_abm_data(abm_file)
        else:
            self.y = []
        # model and obs noise
        self.sigma2_x = sigma2_x
        self.sigma2_y = sigma2_y
        self.theta_beta = jnp.array([-1.0, 1.5, self.sigma2_x])
        self.theta_gamma = jnp.array([-1.5, 0.5, self.sigma2_x])

        self.param_dim = 6
        self.state_dim = 5

    def PX0(self, I0=None):
        """Set prior over initial states for a single sample."""
        S = self.N
        loc = 0.00025
        scale = 0.0005
        lb = 0.0
        ub = 0.001
        lb_trans, ub_trans = (lb - loc) / scale, (ub - loc) / scale
        R = stats.truncnorm(lb_trans, ub_trans, loc=loc, scale=scale).rvs()
        R0 = R * S
        S = S - R0

        I0_key, Y0_key, param_key = jr.split(self.ic_key, 3)

        if I0 is None:
            I0 = self.I0

        I = jnp.log(I0) + jr.uniform(I0_key, minval=-10.0, maxval=-5.5)
        _I0 = jnp.exp(I + jnp.log(S))
        S0 = S - _I0
        # Y0 = jnp.array([S0, _I0, R0])
        Y0 = jnp.array([S0, _I0, R0]) + self.sigma2_y * jr.normal(Y0_key, shape=(3,))
        Y0 = jnp.where(Y0 > 0, Y0, jnp.abs(Y0))
        Y0 = jnp.where(Y0 < 1.0, Y0, 1.0 / Y0)

        # Y0 = Y0 / Y0.sum()

        beta0, gamma0 = jr.uniform(param_key, minval=-5, maxval=1, shape=(2,))

        self.Y0 = jnp.array([Y0[0], Y0[1], Y0[2], beta0, gamma0])
        return self.Y0

    def PX0_rvs(self, size, key):
        """Set prior over initial states for multiple samples."""
        if self.Y0 is not None:
            Y0 = self.Y0.copy()
        else:
            Y0 = self.PX0()
        Y0 = Y0 + self.sigma2_y * jr.normal(key, shape=(size, 5))

        Y0 = jnp.where(Y0 > 0, Y0, jnp.abs(Y0))
        Y0 = jnp.where(Y0 < 1.0, Y0, 1.0 / Y0)
        # tmp = jax.vmap(lambda x: utils.simplex_proj(x, 3), in_axes=0)(Y0[:, :3])
        # Y0 = Y0.at[:, :3].set(tmp)

        return Y0

    def PY(self, t, xp, x, idx, key):
        # taking the abm output + Gaussian noise
        return self.y[idx, 1] + self.sigma2_y * jr.normal(key)

    def PY_rvs(self, t, x, key):
        return x[1] + self.sigma2_y * jr.normal(key)

    @partial(jax.jit, static_argnames=["self"])
    def PY_logpdf(self, t, y, x, key):
        keys = jr.split(key, x.shape[0])
        var = self.sigma2_y + self.sigma2_x
        _ts = jnp.array([0, self.tau])
        _res = jax.vmap(self._solve_sir, in_axes=(None, 0, 0))(_ts, x, keys)
        res = jax.vmap(jstats.norm.logpdf, in_axes=(0, None, None))(
            _res[:, 1], y[1], var
        )
        return res.T

    def PX(self, t, xp, key, add_noise=True):
        """Compute single sample of the push-forward."""
        # make this normal based on \Delta {S, I, R]}
        sir_key, noise_key = jr.split(key, 2)
        _ts = jnp.array([0, self.tau])
        res = self._solve_sir(_ts, xp, sir_key)
        noise = 0.0
        if add_noise:
            noise = jr.normal(noise_key, shape=res.shape) * self.sigma2_x

        return res + noise

    def PX_logpdf(self, t, xp, x):
        """Compute logpdf of SIR model states."""
        print("x", x)
        print("xp", xp)
        # res = jax.vmap()

    def PX_rvs(self, t, xp, key, add_noise=True):
        """Compute an n_particle sample from the push-forward."""
        keys = jr.split(key, xp.shape[0])
        _ts = jnp.array([0, self.tau])
        res = jax.vmap(self._solve_sir, in_axes=(None, 0, 0))(_ts, xp, keys)

        if add_noise:
            res = res.at[:].add(jr.normal(key, shape=res.shape) * self.sigma2_x)
        tmp = jnp.where(res[:, :3] > 0.0, res[:, :3], jnp.abs(res[:, :3]) / 10)
        res = res.at[:, :3].set(tmp)
        tmp = jnp.where(res[:, :3] < 1.0, res[:, :3], 1.0 / res[:, :3])
        res = res.at[:, :3].set(tmp)
        return res

    @partial(jax.jit, static_argnames=["self"])
    def _solve_sir(self, ts, ys, key):
        beta_dbt, gamma_dbt = jr.normal(key, shape=(2,))

        def __sir(t, y, args):
            beta = jnp.exp(y[3])
            gamma = jnp.exp(y[4])
            N, tau, beta_dbt, gamma_dbt, theta_beta, theta_gamma = args

            # beta_dbt, gamma_dbt = jr.normal(key, shape=(2,))

            dS = -(beta * y[0] * y[1])
            dI = (beta * y[0] * y[1]) - y[1] * gamma
            dR = y[1] * gamma

            dlog_beta = (
                theta_beta[0]
                - theta_beta[1] * y[3]
                + (theta_beta[2] * beta_dbt / tau)
            )
            dlog_gamma = (
                theta_gamma[0]
                - theta_gamma[1] * y[4]
                + (theta_gamma[2] * gamma_dbt / tau)
            )
            return jnp.array([dS, dI, dR, dlog_beta, dlog_gamma])

        term = diffrax.ODETerm(__sir)
        solver = diffrax.Tsit5()
        dt0 = 0.1
        saveat = saveat = diffrax.SaveAt(ts=ts)
        args = (self.N, dt0, beta_dbt, gamma_dbt, self.theta_beta, self.theta_gamma)
        sol = diffrax.diffeqsolve(
            term, solver, ts[0], ts[-1], dt0, ys, args=args, saveat=saveat
        )
        return sol.ys[-1]

    def _sir(self, t, y, key, args):
        """Compute deterministic part of sde."""
        beta = jnp.exp(y[3])
        gamma = jnp.exp(y[4])
        N, tau = args

        beta_dbt, gamma_dbt = jr.normal(key, shape=(2,))

        dS = -(beta * y[0] * y[1])
        dI = (beta * y[0] * y[1]) - y[1] * gamma
        dR = y[1] * gamma

        dlog_beta = (
            self.theta_beta[0]
            - self.theta_beta[1] * y[3]
            + (self.theta_beta[2] * beta_dbt / tau)
        )
        # dlog_beta = theta_beta[0] - theta_beta[1] * y[3]
        # dlog_gamma = theta_gamma[0] - theta_gamma[1] * y[4]
        dlog_gamma = (
            self.theta_gamma[0]
            - self.theta_gamma[1] * y[4]
            + (self.theta_gamma[2] * gamma_dbt / tau)
        )
        return jnp.array([dS, dI, dR, dlog_beta, dlog_gamma])

    def set_IC(self, Y0, key):
        Y0 = Y0 + self.sigma2_y * jr.normal(key, shape=(3,))
        # print(Y0.shape)

        print("0", Y0 * 5000)
        tmp = utils.simplex_proj(Y0, 3)
        Y0 = Y0.at[:].set(tmp)
        print("1", 5000 * Y0)
        # Y0 = Y0 / Y0.sum()
        # Y0 = Y0.at[:].set(tmp)
        # print("2",5000* Y0)
        # need to append beta and gamma
        self.Y0 = jnp.array(
            [Y0[0], Y0[1], Y0[2], self.theta_beta[0], self.theta_gamma[0]]
        )
        print("IC", 5000 * self.Y0)
        # print(XXX)

    def update_beta_gamma(self, X):
        mu = X[:, :-2].mean(axis=0)
        sigma = jnp.var(
            X[:, :-2],
            ddof=1,
            axis=0,
        )
        self.theta_beta = self.theta_beta.at[0].set(mu[0])
        # self.theta_beta = self.theta_beta.at[2].set(sigma[0])
        self.theta_gamma = self.theta_gamma.at[0].set(mu[1])
        # self.theta_gamma = self.theta_gamma.at[2].set(sigma[1])

    def update_params(self, theta):
        """Update model params from MCMC."""
        for i in range(3):
            self.theta_beta = self.theta_beta.at[i].set(theta[i])
            self.theta_gamma = self.theta_gamma.at[i].set(theta[i + 3])

    def prior_sample(self, key):
        sample = jnp.zeros(self.param_dim)
        uniform0_key, gamma_key, uniform1_key = jr.split(key, 3)
        _uniform = jr.uniform(uniform0_key, minval=-20, maxval=5, shape=2)
        _gamma = jr.gamma(gamma_key, a=2, shape=2)
        _noise = jr.uniform(uniform1_key, shape=2)
        sample = sample.at[0].set(_uniform[0])
        sample = sample.at[1].set(_gamma[0])
        sample = sample.at[2].set(_noise[0])
        sample = sample.at[3].set(_uniform[1])
        sample = sample.at[4].set(_gamma[1])
        sample = sample.at[5].set(_noise[1])
        return sample

    def eval_log_prior(self, X):
        prior = 0.0
        oob = False
        tmp = jstats.uniform.pdf(X[0], loc=-20, scale=25)
        if tmp == 0.0:
            oob = True
            return (0.0, oob)
        prior += tmp

        tmp = jstats.gamma.pdf(X[1], a=2)
        if tmp == 0.0:
            oob = True
            return (0.0, oob)
        prior += tmp

        tmp = jstats.uniform.pdf(X[2])
        if tmp == 0.0:
            oob = True
            return (0.0, oob)
        prior += tmp

        tmp = jstats.uniform.pdf(X[3], loc=-20, scale=25)
        if tmp == 0.0:
            oob = True
            return (0.0, oob)
        prior += tmp

        tmp = jstats.gamma.pdf(X[4], a=2)
        if tmp == 0.0:
            oob = True
            return (0.0, oob)
        prior += tmp

        tmp = jstats.uniform.pdf(X[5])
        if tmp == 0.0:
            oob = True
            return (0.0, oob)
        prior += tmp
        return (prior, oob)
