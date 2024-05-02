import functools
from jax import numpy as jnp
from jax.scipy.special import digamma
from numpyro import distributions
import operator
from typing import Any
from ..dispatch import valuedispatch
from ..nodes import DelayedValue, Operator


@functools.singledispatch
def expect(self: Any, expr: Any = 1) -> jnp.ndarray:
    raise NotImplementedError


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
    else:
        raise NotImplementedError


@expect.register
def _expect_gamma(self: distributions.Gamma, expr: Any = 1) -> jnp.ndarray:
    if expr == "log":
        return digamma(self.concentration) - jnp.log(self.rate)
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
    else:
        raise NotImplementedError


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
