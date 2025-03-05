# pylint: disable=C0103

from functools import partial

from scipy import stats

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jstats

import diffrax

import utils
from fk import FeynmanKac



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

    def simulate_given_x(self, t, x, key):
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


class SIRSDE(StateSpaceModel):
    def __init__(
        self,
        N,
        I0,
        sigma2_y,
        sigma2_x,
        tau,
        key,
        abm_file=None,
        beta0=None,
        gamma0=None,
    ):
        super().__init__()
        # population size
        self.N = N
        # ICs
        self.I0 = I0
        self.Y0 = None
        # jax random
        self.ic_key, self.obs_key, self.X_key, self.theta_key = jr.split(key, 4)
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
        self.beta0 = beta0
        self.gamma0 = gamma0

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

        Y0_key, param_key = jr.split(self.ic_key, 2)

        if I0 is None:
            I0 = self.I0
        S0 = S - I0

        Y0 = jnp.array([S0, I0, R0]) + (
            0.1 * self.sigma2_y * jr.normal(Y0_key, shape=(3,))
        )
        Y0 = jnp.where(Y0 > 0, Y0, jnp.abs(Y0))
        Y0 = jnp.where(Y0 < 1.0, Y0, 1.0 / Y0)

        if self.beta0 is None:
            self.beta0, self.gamma0 = jr.uniform(
                param_key, minval=-5, maxval=1, shape=(2,)
            )

        self.Y0 = jnp.array([Y0[0], Y0[1], Y0[2], self.beta0, self.gamma0])
        return self.Y0

    def PX0_rvs(self, size, key):
        """Set prior over initial states for multiple samples."""
        if self.Y0 is not None:
            Y0 = self.Y0.copy()
        else:
            Y0 = self.PX0()
        Y0 = Y0 + (0.1 * self.sigma2_y * jr.normal(key, shape=(size, 5)))

        Y0 = Y0.at[:3].set(jnp.where(Y0[:3] > 0, Y0[:3], jnp.abs(Y0[:3])))
        Y0 = Y0.at[:3].set(jnp.where(Y0[:3] < 1.0, Y0[:3], 1.0 / Y0[:3]))

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
        sir_key, noise_key = jr.split(key, 2)
        _ts = jnp.array([0, self.tau])
        res = self._solve_sir(_ts, xp, sir_key)
        noise = 0.0
        if add_noise:
            noise = jr.normal(noise_key, shape=res.shape) * self.sigma2_x

        return res + noise


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

            dS = -(beta * y[0] * y[1])
            dI = (beta * y[0] * y[1]) - y[1] * gamma
            dR = y[1] * gamma

            dlog_beta = (
                theta_beta[0] - theta_beta[1] * y[3] + (theta_beta[2] * beta_dbt / tau)
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
