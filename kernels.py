"""Jax partial implementation of Sk-learn's GP kernels. """
from abc import ABCMeta, abstractmethod
import math
from collections import namedtuple
from inspect import signature


import jax
import jax.numpy as jnp
import numpy as np

from jax.scipy.special import gamma




def _check_length_scale(X, length_scale):
    length_scale = jnp.squeeze(length_scale).astype(float)
    if jnp.ndim(length_scale) > 1:
        raise ValueError("length_scale cannot be of dimension greater than 1")
    if jnp.ndim(length_scale) == 1 and X.shape[1] != length_scale.shape[0]:
        raise ValueError(
            "Anisotropic kernel must have the same number of "
            f"dimensions as data {length_scale.shape[0]!=X.shape[1]}"
        )
    return length_scale


def _pdist_callable(X, *, metric, **kwargs):
    n = X.shape[0]
    out_size = (n * (n - 1)) // 2
    dm = jnp.empty((out_size,), dtype=jnp.float64)
    k = 0
    for i in range(X.shape[0] - 1):
        for j in range(i + 1, X.shape[0]):
            dm[k] = metric(X[i], X[j], **kwargs)
            k += 1
    return dm


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


class Kernel(metaclass=ABCMeta):
    """Base class for all kernels.

    .. versionadded:: 0.18

    Examples
    --------
    >>> from sklearn.gaussian_process.kernels import Kernel, RBF
    >>> import numpy as np
    >>> class CustomKernel(Kernel):
    ...     def __init__(self, length_scale=1.0):
    ...         self.length_scale = length_scale
    ...     def __call__(self, X, Y=None):
    ...         if Y is None:
    ...             Y = X
    ...         return np.inner(X, X if Y is None else Y) ** 2
    ...     def diag(self, X):
    ...         return np.ones(X.shape[0])
    ...     def is_stationary(self):
    ...         return True
    >>> kernel = CustomKernel(length_scale=2.0)
    >>> X = np.array([[1, 2], [3, 4]])
    >>> print(kernel(X))
    [[ 25 121]
     [121 625]]
    """

    def get_params(self, deep=True):
        """Get parameters of this kernel.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = dict()

        # introspect the constructor arguments to find the model parameters
        # to represent
        cls = self.__class__
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        init_sign = signature(init)
        args, varargs = [], []
        for parameter in init_sign.parameters.values():
            if parameter.kind != parameter.VAR_KEYWORD and parameter.name != "self":
                args.append(parameter.name)
            if parameter.kind == parameter.VAR_POSITIONAL:
                varargs.append(parameter.name)

        if len(varargs) != 0:
            raise RuntimeError(
                "scikit-learn kernels should always "
                "specify their parameters in the signature"
                " of their __init__ (no varargs)."
                " %s doesn't follow this convention." % (cls,)
            )
        for arg in args:
            params[arg] = getattr(self, arg)

        return params

    def set_params(self, **params):
        """Set the parameters of this kernel.

        The method works on simple kernels as well as on nested kernels.
        The latter have parameters of the form ``<component>__<parameter>``
        so that it's possible to update each component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        for key, value in params.items():
            split = key.split("__", 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError(
                        "Invalid parameter %s for kernel %s. "
                        "Check the list of available parameters "
                        "with `kernel.get_params().keys()`." % (name, self)
                    )
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if key not in valid_params:
                    raise ValueError(
                        "Invalid parameter %s for kernel %s. "
                        "Check the list of available parameters "
                        "with `kernel.get_params().keys()`."
                        % (key, self.__class__.__name__)
                    )
                setattr(self, key, value)
        return self

    def clone_with_theta(self, theta):
        """Returns a clone of self with given hyperparameters theta.

        Parameters
        ----------
        theta : ndarray of shape (n_dims,)
            The hyperparameters
        """
        cloned = clone(self)
        cloned.theta = theta
        return cloned

    @property
    def n_dims(self):
        """Returns the number of non-fixed hyperparameters of the kernel."""
        return self.theta.shape[0]

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter specifications."""
        r = [
            getattr(self, attr)
            for attr in dir(self)
            if attr.startswith("hyperparameter_")
        ]
        return r

    @property
    def theta(self):
        """Returns the (flattened, log-transformed) non-fixed hyperparameters.

        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.

        Returns
        -------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        theta = []
        params = self.get_params()
        for hyperparameter in self.hyperparameters:
            if not hyperparameter.fixed:
                theta.append(params[hyperparameter.name])
        if len(theta) > 0:
            print(theta[0])
            print(type(theta))
            # return jnp.log(jnp.hstack(theta))
            return np.log(np.hstack(theta))
        else:
            return jnp.array([])

    @theta.setter
    def theta(self, theta):
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters
        ----------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        params = self.get_params()
        i = 0
        for hyperparameter in self.hyperparameters:
            if hyperparameter.fixed:
                continue
            if hyperparameter.n_elements > 1:
                # vector-valued parameter
                params[hyperparameter.name] = jnp.exp(
                    theta[i : i + hyperparameter.n_elements]
                )
                i += hyperparameter.n_elements
            else:
                params[hyperparameter.name] = jnp.exp(theta[i])
                i += 1

        if i != len(theta):
            raise ValueError(
                "theta has not the correct number of entries."
                " Should be %d; given are %d" % (i, len(theta))
            )
        self.set_params(**params)

    @property
    def bounds(self):
        """Returns the log-transformed bounds on the theta.

        Returns
        -------
        bounds : ndarray of shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        bounds = [
            hyperparameter.bounds
            for hyperparameter in self.hyperparameters
            if not hyperparameter.fixed
        ]
        if len(bounds) > 0:
            return np.log(np.vstack(bounds))
        else:
            return np.array([])

    # def __add__(self, b):
    #     if not isinstance(b, Kernel):
    #         return Sum(self, ConstantKernel(b))
    #     return Sum(self, b)

    # def __radd__(self, b):
    #     if not isinstance(b, Kernel):
    #         return Sum(ConstantKernel(b), self)
    #     return Sum(b, self)

    def __mul__(self, b):
        if not isinstance(b, Kernel):
            return Product(self, ConstantKernel(b))
        return Product(self, b)

    # def __rmul__(self, b):
    #     if not isinstance(b, Kernel):
    #         return Product(ConstantKernel(b), self)
    #     return Product(b, self)

    # def __pow__(self, b):
    #     return Exponentiation(self, b)

    def __eq__(self, b):
        if type(self) != type(b):
            return False
        params_a = self.get_params()
        params_b = b.get_params()
        for key in set(list(params_a.keys()) + list(params_b.keys())):
            if np.any(params_a.get(key, None) != params_b.get(key, None)):
                return False
        return True

    def __repr__(self):
        return "{0}({1})".format(
            self.__class__.__name__, ", ".join(map("{0:.3g}".format, self.theta))
        )

    @abstractmethod
    def __call__(self, X, Y=None, eval_gradient=False):
        """Evaluate the kernel."""

    @abstractmethod
    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """

    @abstractmethod
    def is_stationary(self):
        """Returns whether the kernel is stationary."""

    @property
    def requires_vector_input(self):
        """Returns whether the kernel is defined on fixed-length feature
        vectors or generic objects. Defaults to True for backward
        compatibility."""
        return True

    def _check_bounds_params(self):
        """Called after fitting to warn if bounds may have been too tight."""
        list_close = np.isclose(self.bounds, np.atleast_2d(self.theta).T)
        idx = 0
        for hyp in self.hyperparameters:
            if hyp.fixed:
                continue
            for dim in range(hyp.n_elements):
                if list_close[idx, 0]:
                    warnings.warn(
                        "The optimal value found for "
                        "dimension %s of parameter %s is "
                        "close to the specified lower "
                        "bound %s. Decreasing the bound and"
                        " calling fit again may find a "
                        "better value." % (dim, hyp.name, hyp.bounds[dim][0]),
                        ConvergenceWarning,
                    )
                elif list_close[idx, 1]:
                    warnings.warn(
                        "The optimal value found for "
                        "dimension %s of parameter %s is "
                        "close to the specified upper "
                        "bound %s. Increasing the bound and"
                        " calling fit again may find a "
                        "better value." % (dim, hyp.name, hyp.bounds[dim][1]),
                        ConvergenceWarning,
                    )
                idx += 1


class GenericKernelMixin:
    """Mixin for kernels which operate on generic objects such as variable-
    length sequences, trees, and graphs.

    .. versionadded:: 0.22
    """

    @property
    def requires_vector_input(self):
        """Whether the kernel works only on fixed-length feature vectors."""
        return False


class StationaryKernelMixin:
    """Mixin for kernels which are stationary: k(X, Y)= f(X-Y).

    .. versionadded:: 0.18
    """

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return True


class NormalizedKernelMixin:
    """Mixin for kernels which are normalized: k(X, X)=1.

    .. versionadded:: 0.18
    """

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return jnp.ones(X.shape[0])

class KernelOperator(Kernel):
    """Base class for all kernel operators.

    .. versionadded:: 0.18
    """

    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def get_params(self, deep=True):
        """Get parameters of this kernel.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = dict(k1=self.k1, k2=self.k2)
        if deep:
            deep_items = self.k1.get_params().items()
            params.update(("k1__" + k, val) for k, val in deep_items)
            deep_items = self.k2.get_params().items()
            params.update(("k2__" + k, val) for k, val in deep_items)

        return params

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter."""
        r = [
            Hyperparameter(
                "k1__" + hyperparameter.name,
                hyperparameter.value_type,
                hyperparameter.bounds,
                hyperparameter.n_elements,
            )
            for hyperparameter in self.k1.hyperparameters
        ]

        for hyperparameter in self.k2.hyperparameters:
            r.append(
                Hyperparameter(
                    "k2__" + hyperparameter.name,
                    hyperparameter.value_type,
                    hyperparameter.bounds,
                    hyperparameter.n_elements,
                )
            )
        return r

    @property
    def theta(self):
        """Returns the (flattened, log-transformed) non-fixed hyperparameters.

        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.

        Returns
        -------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        return jnp.append(self.k1.theta, self.k2.theta)

    @theta.setter
    def theta(self, theta):
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters
        ----------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        k1_dims = self.k1.n_dims
        self.k1.theta = theta[:k1_dims]
        self.k2.theta = theta[k1_dims:]

    @property
    def bounds(self):
        """Returns the log-transformed bounds on the theta.

        Returns
        -------
        bounds : ndarray of shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        if self.k1.bounds.size == 0:
            return self.k2.bounds
        if self.k2.bounds.size == 0:
            return self.k1.bounds
        return jnp.vstack((self.k1.bounds, self.k2.bounds))

    def __eq__(self, b):
        if type(self) != type(b):
            return False
        return (self.k1 == b.k1 and self.k2 == b.k2) or (
            self.k1 == b.k2 and self.k2 == b.k1
        )

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return self.k1.is_stationary() and self.k2.is_stationary()

    @property
    def requires_vector_input(self):
        """Returns whether the kernel is stationary."""
        return self.k1.requires_vector_input or self.k2.requires_vector_input



class Hyperparameter(
    namedtuple(
        "Hyperparameter", ("name", "value_type", "bounds", "n_elements", "fixed")
    )
):
    """A kernel hyperparameter's specification in form of a namedtuple.

    .. versionadded:: 0.18

    Attributes
    ----------
    name : str
        The name of the hyperparameter. Note that a kernel using a
        hyperparameter with name "x" must have the attributes self.x and
        self.x_bounds

    value_type : str
        The type of the hyperparameter. Currently, only "numeric"
        hyperparameters are supported.

    bounds : pair of floats >= 0 or "fixed"
        The lower and upper bound on the parameter. If n_elements>1, a pair
        of 1d array with n_elements each may be given alternatively. If
        the string "fixed" is passed as bounds, the hyperparameter's value
        cannot be changed.

    n_elements : int, default=1
        The number of elements of the hyperparameter value. Defaults to 1,
        which corresponds to a scalar hyperparameter. n_elements > 1
        corresponds to a hyperparameter which is vector-valued,
        such as, e.g., anisotropic length-scales.

    fixed : bool, default=None
        Whether the value of this hyperparameter is fixed, i.e., cannot be
        changed during hyperparameter tuning. If None is passed, the "fixed" is
        derived based on the given bounds.

    Examples
    --------
    >>> from sklearn.gaussian_process.kernels import ConstantKernel
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import Hyperparameter
    >>> X, y = make_friedman2(n_samples=50, noise=0, random_state=0)
    >>> kernel = ConstantKernel(constant_value=1.0,
    ...    constant_value_bounds=(0.0, 10.0))

    We can access each hyperparameter:

    >>> for hyperparameter in kernel.hyperparameters:
    ...    print(hyperparameter)
    Hyperparameter(name='constant_value', value_type='numeric',
    bounds=array([[ 0., 10.]]), n_elements=1, fixed=False)

    >>> params = kernel.get_params()
    >>> for key in sorted(params): print(f"{key} : {params[key]}")
    constant_value : 1.0
    constant_value_bounds : (0.0, 10.0)
    """

    # A raw namedtuple is very memory efficient as it packs the attributes
    # in a struct to get rid of the __dict__ of attributes in particular it
    # does not copy the string for the keys on each instance.
    # By deriving a namedtuple class just to introduce the __init__ method we
    # would also reintroduce the __dict__ on the instance. By telling the
    # Python interpreter that this subclass uses static __slots__ instead of
    # dynamic attributes. Furthermore we don't need any additional slot in the
    # subclass so we set __slots__ to the empty tuple.
    __slots__ = ()

    def __new__(cls, name, value_type, bounds, n_elements=1, fixed=None):
        if not isinstance(bounds, str) or bounds != "fixed":
            bounds = jnp.atleast_2d(bounds)
            if n_elements > 1:  # vector-valued parameter
                if bounds.shape[0] == 1:
                    bounds = np.repeat(bounds, n_elements, 0)
                elif bounds.shape[0] != n_elements:
                    raise ValueError(
                        "Bounds on %s should have either 1 or "
                        "%d dimensions. Given are %d"
                        % (name, n_elements, bounds.shape[0])
                    )

        if fixed is None:
            fixed = isinstance(bounds, str) and bounds == "fixed"
        return super(Hyperparameter, cls).__new__(
            cls, name, value_type, bounds, n_elements, fixed
        )

    # This is mainly a testing utility to check that two hyperparameters
    # are equal.
    def __eq__(self, other):
        return (
            self.name == other.name
            and self.value_type == other.value_type
            and jnp.all(self.bounds == other.bounds)
            and self.n_elements == other.n_elements
            and self.fixed == other.fixed
        )

class ConstantKernel(StationaryKernelMixin, GenericKernelMixin, Kernel):
    """Constant kernel.

    Can be used as part of a product-kernel where it scales the magnitude of
    the other factor (kernel) or as part of a sum-kernel, where it modifies
    the mean of the Gaussian process.

    .. math::
        k(x_1, x_2) = constant\\_value \\;\\forall\\; x_1, x_2

    Adding a constant kernel is equivalent to adding a constant::

            kernel = RBF() + ConstantKernel(constant_value=2)

    is the same as::

            kernel = RBF() + 2


    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    constant_value : float, default=1.0
        The constant value which defines the covariance:
        k(x_1, x_2) = constant_value

    constant_value_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on `constant_value`.
        If set to "fixed", `constant_value` cannot be changed during
        hyperparameter tuning.

    """

    def __init__(self, constant_value=1.0, constant_value_bounds=(1e-5, 1e5)):
        self.constant_value = constant_value
        self.constant_value_bounds = constant_value_bounds

    @property
    def hyperparameter_constant_value(self):
        return Hyperparameter("constant_value", "numeric", self.constant_value_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Left argument of the returned kernel k(X, Y)

        Y : array-like of shape (n_samples_X, n_features) or list of object, \
            default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
            optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        if Y is None:
            Y = X
        elif eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        K = jnp.full(
            (X.shape[0], Y.shape[0]),
            self.constant_value,
            dtype=jnp.array(self.constant_value).dtype,
        )
        if eval_gradient:
            if not self.hyperparameter_constant_value.fixed:
                return (
                    K,
                    jnp.full(
                        (X.shape[0], X.shape[0], 1),
                        self.constant_value,
                        dtype=jnp.array(self.constant_value).dtype,
                    ),
                )
            else:
                return K, jnp.empty((X.shape[0], X.shape[0], 0))
        else:
            return K

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Argument to the kernel.

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return np.full(
            _num_samples(X),
            self.constant_value,
            dtype=jnp.array(self.constant_value).dtype,
        )

    def __repr__(self):
        return "{0:.3g}**2".format(jnp.sqrt(self.constant_value))



class Product(KernelOperator):
    """The `Product` kernel takes two kernels :math:`k_1` and :math:`k_2`
    and combines them via

    .. math::
        k_{prod}(X, Y) = k_1(X, Y) * k_2(X, Y)

    Note that the `__mul__` magic method is overridden, so
    `Product(RBF(), RBF())` is equivalent to using the * operator
    with `RBF() * RBF()`.

    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    k1 : Kernel
        The first base-kernel of the product-kernel

    k2 : Kernel
        The second base-kernel of the product-kernel


    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import (RBF, Product,
    ...            ConstantKernel)
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = Product(ConstantKernel(2), RBF())
    >>> gpr = GaussianProcessRegressor(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    1.0
    >>> kernel
    1.41**2 * RBF(length_scale=1)
    """

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Left argument of the returned kernel k(X, Y)

        Y : array-like of shape (n_samples_Y, n_features) or list of object,\
            default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        if eval_gradient:
            K1, K1_gradient = self.k1(X, Y, eval_gradient=True)
            K2, K2_gradient = self.k2(X, Y, eval_gradient=True)
            return K1 * K2, np.dstack(
                (K1_gradient * K2[:, :, np.newaxis], K2_gradient * K1[:, :, np.newaxis])
            )
        else:
            return self.k1(X, Y) * self.k2(X, Y)

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Argument to the kernel.

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return self.k1.diag(X) * self.k2.diag(X)

    def __repr__(self):
        return "{0} * {1}".format(self.k1.constant_value, self.k2)





class Kernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """Base class for all kernels.

    .. versionadded:: 0.18

    Examples
    --------
    >>> from sklearn.gaussian_process.kernels import Kernel, RBF
    >>> import numpy as np
    >>> class CustomKernel(Kernel):
    ...     def __init__(self, length_scale=1.0):
    ...         self.length_scale = length_scale
    ...     def __call__(self, X, Y=None):
    ...         if Y is None:
    ...             Y = X
    ...         return np.inner(X, X if Y is None else Y) ** 2
    ...     def diag(self, X):
    ...         return np.ones(X.shape[0])
    ...     def is_stationary(self):
    ...         return True
    >>> kernel = CustomKernel(length_scale=2.0)
    >>> X = np.array([[1, 2], [3, 4]])
    >>> print(kernel(X))
    [[ 25 121]
     [121 625]]
    """

    def get_params(self, deep=True):
        """Get parameters of this kernel.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = dict()

        # introspect the constructor arguments to find the model parameters
        # to represent
        cls = self.__class__
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        init_sign = signature(init)
        args, varargs = [], []
        for parameter in init_sign.parameters.values():
            if parameter.kind != parameter.VAR_KEYWORD and parameter.name != "self":
                args.append(parameter.name)
            if parameter.kind == parameter.VAR_POSITIONAL:
                varargs.append(parameter.name)

        if len(varargs) != 0:
            raise RuntimeError(
                "scikit-learn kernels should always "
                "specify their parameters in the signature"
                " of their __init__ (no varargs)."
                " %s doesn't follow this convention." % (cls,)
            )
        for arg in args:
            params[arg] = getattr(self, arg)

        return params

    def set_params(self, **params):
        """Set the parameters of this kernel.

        The method works on simple kernels as well as on nested kernels.
        The latter have parameters of the form ``<component>__<parameter>``
        so that it's possible to update each component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        for key, value in params.items():
            split = key.split("__", 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError(
                        "Invalid parameter %s for kernel %s. "
                        "Check the list of available parameters "
                        "with `kernel.get_params().keys()`." % (name, self)
                    )
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if key not in valid_params:
                    raise ValueError(
                        "Invalid parameter %s for kernel %s. "
                        "Check the list of available parameters "
                        "with `kernel.get_params().keys()`."
                        % (key, self.__class__.__name__)
                    )
                setattr(self, key, value)
        return self

    def clone_with_theta(self, theta):
        """Returns a clone of self with given hyperparameters theta.

        Parameters
        ----------
        theta : ndarray of shape (n_dims,)
            The hyperparameters
        """
        cloned = clone(self)
        cloned.theta = theta
        return cloned

    @property
    def n_dims(self):
        """Returns the number of non-fixed hyperparameters of the kernel."""
        return self.theta.shape[0]

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter specifications."""
        r = [
            getattr(self, attr)
            for attr in dir(self)
            if attr.startswith("hyperparameter_")
        ]
        return r

    @property
    def theta(self):
        """Returns the (flattened, log-transformed) non-fixed hyperparameters.

        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.

        Returns
        -------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        theta = []
        params = self.get_params()
        for hyperparameter in self.hyperparameters:
            if not hyperparameter.fixed:
                theta.append(params[hyperparameter.name])
        theta = np.array(theta)
        if theta.shape[0] > 0:
            return np.log(theta.reshape(-1, 1))
        else:
            return np.array([])

    @theta.setter
    def theta(self, theta):
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters
        ----------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        params = self.get_params()
        i = 0
        for hyperparameter in self.hyperparameters:
            if hyperparameter.fixed:
                continue
            if hyperparameter.n_elements > 1:
                # vector-valued parameter
                params[hyperparameter.name] = jnp.exp(
                    theta[i : i + hyperparameter.n_elements]
                )
                i += hyperparameter.n_elements
            else:
                params[hyperparameter.name] = jnp.exp(theta[i])
                i += 1

        if i != len(theta):
            raise ValueError(
                "theta has not the correct number of entries."
                " Should be %d; given are %d" % (i, len(theta))
            )
        self.set_params(**params)

    @property
    def bounds(self):
        """Returns the log-transformed bounds on the theta.

        Returns
        -------
        bounds : ndarray of shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        bounds = [
            hyperparameter.bounds
            for hyperparameter in self.hyperparameters
            if not hyperparameter.fixed
        ]
        if len(bounds) > 0:
            return jnp.log(jnp.vstack(bounds))
        else:
            return jnp.array([])


class RBF(Kernel):
    """Radial basis function kernel (aka squared-exponential kernel).

    The RBF kernel is a stationary kernel. It is also known as the
    "squared exponential" kernel. It is parameterized by a length scale
    parameter :math:`l>0`, which can either be a scalar (isotropic variant
    of the kernel) or a vector with the same number of dimensions as the inputs
    X (anisotropic variant of the kernel). The kernel is given by:

    .. math::
        k(x_i, x_j) = \\exp\\left(- \\frac{d(x_i, x_j)^2}{2l^2} \\right)

    where :math:`l` is the length scale of the kernel and
    :math:`d(\\cdot,\\cdot)` is the Euclidean distance.
    For advice on how to set the length scale parameter, see e.g. [1]_.

    This kernel is infinitely differentiable, which implies that GPs with this
    kernel as covariance function have mean square derivatives of all orders,
    and are thus very smooth.
    See [2]_, Chapter 4, Section 4.2, for further details of the RBF kernel.

    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    length_scale : float or ndarray of shape (n_features,), default=1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.

    length_scale_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'length_scale'.
        If set to "fixed", 'length_scale' cannot be changed during
        hyperparameter tuning.

    References
    ----------
    .. [1] `David Duvenaud (2014). "The Kernel Cookbook:
        Advice on Covariance functions".
        <https://www.cs.toronto.edu/~duvenaud/cookbook/>`_

    .. [2] `Carl Edward Rasmussen, Christopher K. I. Williams (2006).
        "Gaussian Processes for Machine Learning". The MIT Press.
        <http://www.gaussianprocess.org/gpml/>`_


    """

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def anisotropic(self):
        return jnp.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter(
                "length_scale",
                "numeric",
                self.length_scale_bounds,
                len(self.length_scale),
            )
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)



    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        """
        X = jnp.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            dists = pdist(X / length_scale)
            K = jnp.exp(-0.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = jnp.fill_diagonal(K, 1, inplace=False)
        else:
            dists = cdist(X / length_scale, Y / length_scale)
            K = jnp.exp(-0.5 * dists)


        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, jnp.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = (X[:, jnp.newaxis, :] - X[jnp.newaxis, :, :]) ** 2 / (
                    length_scale**2
                )
                K_gradient *= K[..., jnp.newaxis]
                return K, K_gradient
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}])".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self.length_scale)),
            )
        else:  # isotropic
            return "{0}(length_scale={1:.3g})".format(
                self.__class__.__name__, jnp.ravel(self.length_scale)[0]
            )


class Matern(RBF):
    """Matern kernel.

    The class of Matern kernels is a generalization of the :class:`RBF`.
    It has an additional parameter :math:`\\nu` which controls the
    smoothness of the resulting function. The smaller :math:`\\nu`,
    the less smooth the approximated function is.
    As :math:`\\nu\\rightarrow\\infty`, the kernel becomes equivalent to
    the :class:`RBF` kernel. When :math:`\\nu = 1/2`, the Matérn kernel
    becomes identical to the absolute exponential kernel.
    Important intermediate values are
    :math:`\\nu=1.5` (once differentiable functions)
    and :math:`\\nu=2.5` (twice differentiable functions).

    The kernel is given by:

    .. math::
         k(x_i, x_j) =  \\frac{1}{\\Gamma(\\nu)2^{\\nu-1}}\\Bigg(
         \\frac{\\sqrt{2\\nu}}{l} d(x_i , x_j )
         \\Bigg)^\\nu K_\\nu\\Bigg(
         \\frac{\\sqrt{2\\nu}}{l} d(x_i , x_j )\\Bigg)



    where :math:`d(\\cdot,\\cdot)` is the Euclidean distance,
    :math:`K_{\\nu}(\\cdot)` is a modified Bessel function and
    :math:`\\Gamma(\\cdot)` is the gamma function.
    See [1]_, Chapter 4, Section 4.2, for details regarding the different
    variants of the Matern kernel.

    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    length_scale : float or ndarray of shape (n_features,), default=1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.

    length_scale_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'length_scale'.
        If set to "fixed", 'length_scale' cannot be changed during
        hyperparameter tuning.

    nu : float, default=1.5
        The parameter nu controlling the smoothness of the learned function.
        The smaller nu, the less smooth the approximated function is.
        For nu=inf, the kernel becomes equivalent to the RBF kernel and for
        nu=0.5 to the absolute exponential kernel. Important intermediate
        values are nu=1.5 (once differentiable functions) and nu=2.5
        (twice differentiable functions). Note that values of nu not in
        [0.5, 1.5, 2.5, inf] incur a considerably higher computational cost
        (appr. 10 times higher) since they require to evaluate the modified
        Bessel function. Furthermore, in contrast to l, nu is kept fixed to
        its initial value and not optimized.

    References
    ----------
    .. [1] `Carl Edward Rasmussen, Christopher K. I. Williams (2006).
        "Gaussian Processes for Machine Learning". The MIT Press.
        <http://www.gaussianprocess.org/gpml/>`_

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.gaussian_process import GaussianProcessClassifier
    >>> from sklearn.gaussian_process.kernels import Matern
    >>> X, y = load_iris(return_X_y=True)
    >>> kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
    >>> gpc = GaussianProcessClassifier(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpc.score(X, y)
    0.9866...
    >>> gpc.predict_proba(X[:2,:])
    array([[0.8513..., 0.0368..., 0.1117...],
            [0.8086..., 0.0693..., 0.1220...]])
    """

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), nu=1.5):
        super().__init__(length_scale, length_scale_bounds)
        self.nu = nu

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        X = jnp.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            dists = pdist(X / length_scale, sqrt=True)
        else:
            dists = cdist(X / length_scale, Y / length_scale, sqrt=True)

        if self.nu == 0.5:
            K = jnp.exp(-dists)
        elif self.nu == 1.5:
            K = dists * math.sqrt(3)
            K = (1.0 + K) * jnp.exp(-K)
        elif self.nu == 2.5:
            K = dists * math.sqrt(5)
            K = (1.0 + K + K**2 / 3.0) * jnp.exp(-K)
        elif self.nu == jnp.inf:
            K = jnp.exp(-(dists**2) / 2.0)
        # else:  # general case; expensive to evaluate
        #     K = dists
        #     K[K == 0.0] += jnp.finfo(float).eps  # strict zeros result in nan
        #     tmp = math.sqrt(2 * self.nu) * K
        #     K.fill((2 ** (1.0 - self.nu)) / gamma(self.nu))
        #     K *= tmp**self.nu
        #     K *= kv(self.nu, tmp)  # Bessel function, see scipy.special.kv

        if Y is None:
            # convert from upper-triangular matrix to square matrix
            K = jnp.fill_diagonal(K, 1, inplace=False)

        return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}], nu={2:.3g})".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self.length_scale)),
                self.nu,
            )
        else:
            return "{0}(length_scale={1:.3g}, nu={2:.3g})".format(
                self.__class__.__name__, jnp.ravel(self.length_scale)[0], self.nu
            )
