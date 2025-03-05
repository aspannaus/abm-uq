



class Summaries:
    """Class to store and update summaries.

    Attribute ``summaries`` of ``SMC`` objects is an instance of this class.
    """

    def __init__(self, cols):
        self._collectors = [cls() for cls in default_collector_cls]
        if cols is not None:
            # call each collector to get a fresh instance
            self._collectors.extend(col() for col in cols)
        for col in self._collectors:
            setattr(self, col.summary_name, col.summary)

    def collect(self, smc):
        for col in self._collectors:
            col.collect(smc)

    def clear_history(
        self,
    ):
        self._collectors.clear()
        self._collectors = [cls() for cls in default_collector_cls]


class Collector:
    """Base class for collectors.

    To subclass `Collector`:

    * implement method `fetch(self, smc)` which computes the summary that
      must be collected (from object smc, at each time).
    * (optionally) define class attribute `summary_name` (name of the collected summary;
      by default, name of the class, un-capitalised, i.e. Moments > moments)
    * (optionally) define class attribute `signature` (the signature of the
      constructor, by default, an empty dict)
    """

    signature = {}

    @property
    def summary_name(self):
        cn = self.__class__.__name__
        return cn[0].lower() + cn[1:]

    def __init__(self, **kwargs):
        self.summary = []
        for k, v in self.signature.items():
            setattr(self, k, v)
        for k, v in kwargs.items():
            if k in self.signature.keys():
                setattr(self, k, v)
            else:
                raise ValueError(
                    f"Collector {self.__class__.__name__}: unknown parameter {k}"
                )

    def __call__(self):
        # clone the object
        return self.__class__(**{k: getattr(self, k) for k in self.signature.keys()})

    def collect(self, smc):
        self.summary.append(self.fetch(smc))


# Default collectors
####################


class ESSs(Collector):
    summary_name = "ESSs"

    def fetch(self, smc):
        return smc.wgts.ESS


class LogLts(Collector):
    def fetch(self, smc):
        return smc.logLt


class Rs_flags(Collector):
    def fetch(self, smc):
        return smc.rs_flag


default_collector_cls = [ESSs, LogLts, Rs_flags]

# Moments
#########


class Moments(Collector):
    """Collects empirical moments (e.g. mean and variance) of the particles.

    Moments are defined through a function phi with the following signature:

        def mom_func(W, X):
           return np.average(X, weights=W)  # for instance

    If no function is provided, the default moment of the Feynman-Kac class
    is used (mean and variance of the particles, see ``core.FeynmanKac``).
    """

    signature = {"mom_func": None}

    def fetch(self, smc):
        f = smc.fk.default_moments if self.mom_func is None else self.mom_func
        return f(smc.wgts.W, smc.X)
