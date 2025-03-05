from collections import deque

import jax.numpy as jnp




def generate_hist_obj(option, smc, key):
    if option is True:
        return ParticleHistory(smc.fk, key)
    elif option is False:
        return None
    elif callable(option):
        return PartialParticleHistory(option)
    elif isinstance(option, int) and option >= 0:
        return RollingParticleHistory(option)
    else:
        raise ValueError("store_history: invalid option")


class PartialParticleHistory:
    """Partial history.

    History that records the particle system only at certain times.
    See `smoothing` module doc for more details.
    """

    def __init__(self, func):
        self.is_save_time = func
        self.X, self.wgts = {}, {}

    def save(self, smc):
        t = smc.t
        if self.is_save_time(t):
            self.X[t] = smc.X
            self.wgts[t] = smc.wgts


class RollingParticleHistory:
    """Rolling window history.

    History that keeps only the k most recent particle systems. Based on
    deques. See `smoothing` module doc for more details.

    """

    def __init__(self, length):
        self.X = deque([], length)
        self.A = deque([], length)
        self.wgts = deque([], length)

    @property
    def N(self):
        """Number of particles at each time step."""
        return self.X[0].shape[0]

    @property
    def T(self):
        """Current length of history."""
        return len(self.X)

    def save(self, smc):
        self.X.append(smc.X)
        self.A.append(smc.A)
        self.wgts.append(smc.wgts)

    def compute_trajectories(self):
        """Compute the N trajectories that constitute the current genealogy.

        Returns a (T, N) int array, such that B[t, n] is the index of ancestor
        at time t of particle X_T^n, where T is the current length of history.
        """
        Bs = [jnp.arange(self.N)]
        for A in list(self.A)[-1:0:-1]:  # list in case self.A is a deque
            Bs.append(A[Bs[-1]])
        Bs.reverse()
        return jnp.array(Bs)


class ParticleHistory(RollingParticleHistory):
    """Particle history.

    A class to store the full history of a particle algorithm, i.e.
    at each time t=0,...T, the N particles, their weights, and their ancestors.
    Off-line smoothing algorithms are methods of this class.

    `SMC` creates an object of this class when invoked with
    ``store_history=True``, and then save at every time t the set of particles,
    their weights (and their logarithm), and the ancestor variables.

    Attributes
    ----------
    X : list
        X[t] is the object that represents the N particles at iteration t
    wgts : list
        wgts[t] is an array of log weighrs that represents at time t
    A : list
        A[t] is the vector of ancestor indices at time t

    """

    def __init__(self, fk, key):
        self.X, self.A, self.wgts = [], [], []
        self.fk = fk
        self.key = key

    def save(self, smc):
        self.X.append(smc.X)
        self.A.append(smc.A)
        self.wgts.append(smc.wgts.lw)
