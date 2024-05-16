import functools
import ifnt
from jax import numpy as jnp
from jax.scipy.special import digamma
from numpyro import distributions
from typing import Any, Union
from ..distributions import LinearDynamicalSystem, PrecisionNormal, Reshaped
from ..nodes import (
    AddOperator,
    DelayedValue,
    GetItemOperator,
    MatMulOperator,
    MulOperator,
    SumOperator,
)
from ..util import multidigamma


@functools.singledispatch
def expect(self: Any, expr: Any = 1) -> jnp.ndarray:
    raise NotImplementedError(self, expr)


@expect.register
def _expected_delayed_value(self: DelayedValue, expr: Any = 1) -> jnp.ndarray:
    if self.value is None:
        raise ValueError(f"Delayed value named '{self.name}' does not have a value.")
    return expect(self.value, expr)


@expect.register(distributions.Normal)
@expect.register(PrecisionNormal)
def _expect_normal(self: distributions.Distribution, expr: Any = 1) -> jnp.ndarray:
    if expr == "exp":
        return jnp.exp(self.mean + self.variance / 2)
    return _expect_distribution(self, expr)


@expect.register
def _expect_independent(self: distributions.Independent, expr: Any = 1) -> jnp.ndarray:
    if expr == "exp":
        return expect(self.base_dist, expr)
    return _expect_distribution(self, expr)


@expect.register
def _expect_distribution(
    self: distributions.Distribution, expr: Any = 1
) -> jnp.ndarray:
    """
    Generic expectations for distributions. We do not register the base class
    :class:`numpyro.distributions.Distribution` to avoid accidental incorrect results
    for distributions with different properties. Further distributions can be registered
    by calling :code:`expect.register(cls, _expect_distribution)`.
    """
    if expr == 1:
        return self.mean
    elif expr == 2:
        return jnp.square(self.mean) + self.variance
    elif expr == "var":
        return self.variance
    elif expr == "outer":
        # For a distribution with `batch_shape = xxx` and `event_shape = yyy`, the
        # "outer" expectation should have shape `xxx + yyy + yyy`. We raise an error for
        # "outer" expectations of distributions without event dimensions.
        assert (
            self.event_dim
        ), f"Expected at least one event dimension; got {self.event_dim}."
        mean: jnp.ndarray = self.mean
        ones = (1,) * self.event_dim
        left = mean.reshape(self.batch_shape + self.event_shape + ones)
        right = mean.reshape(self.batch_shape + ones + self.event_shape)
        outer_mean = left * right

        # Add on the variance.
        eye = None
        for size in self.event_shape:
            if eye is None:
                eye = jnp.eye(size)
            else:
                eye = eye[..., :, None, :, None] * jnp.eye(size)[None, :, None, :]
        outer_var = (
            self.variance.reshape(self.batch_shape + ones + self.event_shape) * eye
        )

        return outer_mean + outer_var
    else:
        raise NotImplementedError(self, expr)


@expect.register
def _expect_delta(self: distributions.Delta, expr: Any = 1) -> jnp.ndarray:
    if expr == "log":
        return jnp.log(self.v)
    elif expr == "logabsdet":
        return jnp.linalg.slogdet(self.v).logabsdet
    elif expr == "exp":
        return jnp.exp(self.v)
    elif callable(expr):
        return expr(self.v)
    else:
        return _expect_distribution(self, expr)


@expect.register
def _expect_gamma(self: distributions.Gamma, expr: Any = 1) -> jnp.ndarray:
    if expr == "log":
        return digamma(self.concentration) - jnp.log(self.rate)
    else:
        return _expect_distribution(self, expr)


@expect.register
def _expect_wishart(self: distributions.Wishart, expr: Any = 1) -> jnp.ndarray:
    if expr == "logabsdet":
        # https://en.wikipedia.org/wiki/Wishart_distribution#Log-expectation
        p = self.scale_tril.shape[-1]
        half_logabsdet = jnp.linalg.slogdet(self.scale_tril).logabsdet
        return (
            multidigamma(self.concentration / 2, p)
            + p * jnp.log(2)
            + 2 * half_logabsdet
        )
    else:
        return _expect_distribution(self, expr)


@expect.register(jnp.ndarray)
@expect.register(int)
@expect.register(float)
def _expect_literal(self, expr=1):
    if expr == 1:
        return self
    elif expr == 2:
        return jnp.square(self)
    elif expr == "log":
        return jnp.log(self)
    elif expr == "outer":
        raise ValueError(
            "Cannot evaluate the outer expectation of a literal value because the "
            "event shape is ambiguous. Wrap the value in a `Delta` distribution and "
            "specify the number of event dimensions to evaluate the outer expectation."
        )
    elif expr == "logabsdet":
        return jnp.linalg.slogdet(self).logabsdet
    elif expr == "exp":
        return jnp.exp(self)
    elif callable(expr):
        return expr(self)
    else:
        raise NotImplementedError(self, expr)


@expect.register
def _expect_reshaped(self: Reshaped, expr: Any = 1) -> jnp.ndarray:
    base = expect(self.base_dist, expr)
    shape = self.shape()
    # For the outer expectation, we need to do some more careful reshaping.
    if expr == "outer":
        shape = shape + self.event_shape
    return base.reshape(shape)


@expect.register(distributions.LowRankMultivariateNormal)
@expect.register(distributions.MultivariateNormal)
def _expect_multivariate_normal(
    self: Union[
        distributions.LowRankMultivariateNormal, distributions.MultivariateNormal
    ],
    expr: Any = 1,
) -> jnp.ndarray:
    if expr == "outer":
        return (
            self.mean[..., :, None] * self.mean[..., None, :] + self.covariance_matrix
        )
    if expr == "exp":
        # https://en.wikipedia.org/wiki/Log-normal_distribution#Multivariate_log-normal
        return jnp.exp(
            self.loc + jnp.diagonal(self.covariance_matrix, axis1=-1, axis2=-2) / 2
        )
    else:
        return _expect_distribution(self, expr)


@expect.register
def _expect_linear_dynamical_system(
    self: LinearDynamicalSystem, expr: Any = 1
) -> jnp.ndarray:
    if expr == "outer":
        # Slightly verbose, but useful for line-profiling.
        mean2 = self.mean[..., :, :, None, None] * self.mean[..., None, None, :, :]
        cov = self.covariance_tensor
        return mean2 + cov
    else:
        return _expect_distribution(self, expr)


@expect.register
def _expect_operator_matmul(self: MatMulOperator, expr: Any = 1) -> jnp.ndarray:
    # FIXME: We only support a design matrix and coefficient vector but complain
    # extensively to avoid unexpected behavior.
    a, b = DelayedValue.materialize(*self.args)
    if isinstance(a, jnp.ndarray):
        assert a.ndim == 2
        a = distributions.Delta(a, event_dim=1)
    assert a.event_dim == 1, "Design matrix must have one event dimension."

    if isinstance(b, jnp.ndarray):
        assert b.ndim == 1
        b = distributions.Delta(b, event_dim=1)
    assert b.event_dim == 1, "Coefficient vector must have one event dimension."

    assert a.event_shape == b.event_shape, "Event shapes must match."

    if expr == 1:
        return expect(a) @ expect(b)
    elif expr == 2:
        return (expect(a, "outer") * expect(b, "outer")).sum(axis=(-1, -2))
    elif expr == "var":
        return _expect_operator_matmul(self, 2) - jnp.square(
            _expect_operator_matmul(self)
        )
    elif expr == "exp":
        # FIXME: This only works for a and b having independent elements.
        ma = expect(a)
        mb = expect(b)
        va = expect(a, "var")
        vb = expect(b, "var")
        norm = 1 - va * vb
        arg = (2 * ma * mb + mb**2 * va + ma**2 * vb) / (2 * norm)
        return (jnp.exp(arg) / jnp.sqrt(norm)).prod(axis=-1)
    else:
        raise NotImplementedError(self, expr)


@expect.register
def _expect_operator_add(self: AddOperator, expr: Any = 1) -> jnp.ndarray:
    if expr == 1:
        return sum(expect(arg) for arg in self.args)
    elif expr == 2:
        return jnp.square(sum(expect(arg) for arg in self.args)) + sum(
            expect(arg, expr="var") for arg in self.args
        )
    elif expr == "var":
        return sum(expect(arg, expr="var") for arg in self.args)
    elif expr == "exp":
        value = 1.0
        for arg in self.args:
            value = value * expect(arg, "exp")
        return value
    else:
        raise NotImplementedError(self, expr)


@expect.register
def _expect_operator_getitem(self: GetItemOperator, expr: Any = 1) -> jnp.ndarray:
    arg, key = self.args
    return ifnt.index_guard(expect(arg, expr))[key]


@expect.register
def _expect_operator_sum(self: SumOperator, expr: Any = 1) -> jnp.ndarray:
    if expr == 1:
        return expect(self.args[0], expr).sum(**self.kwargs)
    elif expr == 2:
        return expect(self, 1)**2 + expect(self, "var")
    elif expr == "var":
        assert isinstance(
            self.kwargs["axis"], int
        ), "Only implemented sum for aggregation over one axis."
        return expect(self.args[0], "var").sum(**self.kwargs)
    raise NotImplementedError(self, expr)


@expect.register
def _expect_operator_mul(self: MulOperator, expr: Any = 1) -> jnp.ndarray:
    a, b = self.args
    if expr == 1:
        # Assuming that a and b are independent.
        return expect(a) * expect(b)
    elif expr == 2:
        return expect(a, expr) * expect(b, expr)
    elif expr == "var":
        return expect(self, 2) - expect(self) ** 2
    raise NotImplementedError(self, expr)
