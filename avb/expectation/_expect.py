import functools
from jax import numpy as jnp
from jax.scipy.special import digamma
from numpyro import distributions
import operator
from typing import Any
from ..dispatch import valuedispatch
from ..distributions import Reshaped
from ..nodes import DelayedValue, Operator
from ..util import multidigamma


@functools.singledispatch
def expect(self: Any, expr: Any = 1) -> jnp.ndarray:
    raise NotImplementedError(self, expr)


@expect.register
def _expected_delayed_value(self: DelayedValue, expr: Any = 1) -> jnp.ndarray:
    if self.value is None:
        raise ValueError(f"Delayed value named '{self.name}' does not have a value.")
    return expect(self.value, expr)


@expect.register
def _expect_distribution(
    self: distributions.Distribution, expr: Any = 1
) -> jnp.ndarray:
    if expr == 1:
        return self.mean
    elif expr == 2:
        return jnp.square(self.mean) + self.variance
    elif expr == "var":
        return self.variance
    elif expr == "outer":
        # For a distribution with `batch_shape = xxx` and `event_shape = yyy`, the
        # "outer" expectation should have shape `xxx + yyy + yyy`.
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
    elif expr == "logabsdet":
        return jnp.linalg.slogdet(self).logabsdet
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


@expect.register
def _expect_operator_node(self: Operator, expr: Any = 1) -> jnp.ndarray:
    # Dispatch from an operator instance to the specific operation we're evaluating.
    return _expect_operator(self, expr)


@valuedispatch(key=lambda x: x.operator)
def _expect_operator(self: Operator, expr: Any = 1) -> jnp.ndarray:
    raise NotImplementedError


@_expect_operator.register(operator.matmul)
def _expect_operator_matmul(self: Operator, expr: Any = 1) -> jnp.ndarray:
    # FIXME: We're making some bold independence assumptions for the second moment and
    # variance.
    a, b = self.args
    if expr == 1:
        return expect(a) @ expect(b)
    elif expr == 2:
        return expect(a, 2) @ expect(b, 2)
    elif expr == "var":
        return expect(a, 2) @ expect(b, 2) - jnp.square(expect(a) @ expect(b))
    else:
        raise NotImplementedError


@_expect_operator.register(operator.add)
def _expect_operator_add(self: Operator, expr: Any = 1) -> jnp.ndarray:
    if expr == 1:
        return sum(expect(arg) for arg in self.args)
    elif expr == 2:
        return jnp.square(sum(expect(arg) for arg in self.args)) + sum(
            expect(arg, expr="var") for arg in self.args
        )
    elif expr == "var":
        return sum(expect(arg, expr="var") for arg in self.args)
    else:
        raise NotImplementedError


@_expect_operator.register(operator.getitem)
def _expect_operator_getitem(self: Operator, expr: Any = 1) -> jnp.ndarray:
    arg, key = self.args
    return expect(arg, expr)[key]
