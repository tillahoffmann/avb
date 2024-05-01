import functools
import inspect
from jax import numpy as jnp
from jax.scipy.special import digamma
from numpyro import distributions
import operator
from typing import Any, Callable
from .dispatch import valuedispatch
from .nodes import LazySample, Operator


def apply_substitutions(func):
    """
    Substitute arguments and keyword arguments with concrete values where available.
    """

    signature = inspect.signature(func)
    param = signature.parameters.get("substitutions")
    assert (
        param and param.kind == param.KEYWORD_ONLY
    ), f"`{func}` must have a keyword-only argument `substitutions`."

    @functools.wraps(func)
    def _apply_substitutions_wrapper(*args, **kwargs):
        substitutions = kwargs.get("substitutions", {})
        args = [
            substitutions[arg] if isinstance(arg, LazySample) else arg for arg in args
        ]
        kwargs = {
            key: substitutions[arg] if isinstance(arg, LazySample) else arg
            for key, arg in kwargs.items()
        }
        return func(*args, **kwargs)

    return _apply_substitutions_wrapper


@functools.singledispatch
def expect(self: Any, expr: Any = 1, *, substitutions=None) -> jnp.ndarray:
    raise NotImplementedError


@expect.register
def _expect_distribution(
    self: distributions.Distribution, expr: Any = 1, *, substitutions=None
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
def _expect_gamma(
    self: distributions.Gamma, expr: Any = 1, *, substitutions=None
) -> jnp.ndarray:
    if expr == "log":
        return digamma(self.concentration) - jnp.log(self.rate)
    else:
        return _expect_distribution(self, expr)


@expect.register(jnp.ndarray)
@expect.register(int)
@expect.register(float)
def _expect_literal(self, expr=1, *, substitutions=None):
    if expr == 1:
        return self
    elif expr == 2:
        return jnp.square(self)
    elif expr == "log":
        return jnp.log(self)
    else:
        raise NotImplementedError


@expect.register
def _expect_operator(
    self: Operator, expr: Any = 1, *, substitutions=None
) -> jnp.ndarray:
    # Dispatch from an operator instance to the specific operation we're evaluating.
    return _expect_unpacked_operator(
        self.operation,
        *self.args,
        **self.kwargs,
        expr=expr,
        substitutions=substitutions,
    )


@valuedispatch
def _expect_unpacked_operator(
    operation: Callable, *args, expr: Any, substitutions=None, **kwargs
) -> jnp.ndarray:
    raise NotImplementedError


@_expect_unpacked_operator.register(operator.matmul)
@apply_substitutions
def _expect_operation_matmul(
    operation, a, b, expr, *, substitutions=None
) -> jnp.ndarray:
    subexpect = functools.partial(expect, substitutions=substitutions)
    if expr == 1:
        return subexpect(a) @ subexpect(b)
    elif expr == 2:
        # TODO: implement this more generally. We currently only support a fixed design
        # matrix `a` on the left and stochastic vector `b` on the right. Batching isn't
        # supported.
        assert isinstance(a, jnp.ndarray)
        assert isinstance(b, jnp.ndarray) or (
            isinstance(b, distributions.Distribution) and b.event_dim == 0
        )
        return jnp.square(a) @ subexpect(b, 2)
    elif expr == "var":
        return jnp.square(a) @ subexpect(b, 2) - jnp.square(subexpect(a) @ subexpect(b))
    else:
        raise NotImplementedError


@_expect_unpacked_operator.register(operator.add)
@apply_substitutions
def _expect_unpacked_operation_add(
    operation, *args, expr, substitutions=None
) -> jnp.ndarray:
    subexpect = functools.partial(expect, substitutions=substitutions)
    if expr == 1:
        return sum(subexpect(arg) for arg in args)
    elif expr == 2:
        return jnp.square(sum(subexpect(arg) for arg in args)) + sum(
            subexpect(arg, expr="var") for arg in args
        )
    elif expr == "var":
        return sum(subexpect(arg, expr="var") for arg in args)
    else:
        raise NotImplementedError
