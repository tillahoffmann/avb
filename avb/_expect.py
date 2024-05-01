import functools
from jax import numpy as jnp
from jax.scipy.special import digamma
from numpyro import distributions
import operator
from typing import Any
from .dispatch import reraise_not_implemented_with_args, valuedispatch
from .nodes import Operator


def apply_substitutions(func):
    """
    Substitute arguments and keyword arguments with concrete values where available.
    """
    from .nodes import LazySample

    @functools.wraps(func)
    def _apply_substitutions_wrapper(*args, **kwargs):
        substitutions = kwargs.pop("substitutions", {})
        args = [
            substitutions[arg] if isinstance(arg, LazySample) else arg for arg in args
        ]
        kwargs = {
            key: substitutions[arg] if isinstance(arg, LazySample) else arg
            for key, arg in kwargs.items()
        }
        return func(*args, **kwargs, substitutions=substitutions)

    return _apply_substitutions_wrapper


@functools.singledispatch
@reraise_not_implemented_with_args
def expect(self: Any, expr: Any = 1, *, substitutions=None) -> jnp.ndarray:
    raise NotImplementedError


@expect.register
@apply_substitutions
def _expect_distribution(
    self: distributions.Distribution, expr: Any = 1, *, substitutions=None
) -> jnp.ndarray:
    if expr == 1:
        return self.mean
    elif expr == 2:
        return jnp.square(self.mean) + self.variance
    else:
        raise NotImplementedError


@expect.register
@apply_substitutions
def _expect_gamma(
    self: distributions.Gamma, expr: Any = 1, *, substitutions=None
) -> jnp.ndarray:
    if expr == "log":
        return digamma(self.concentration) - jnp.log(self.rate)
    else:
        return _expect_distribution(self, expr, substitutions=substitutions)


@expect.register(jnp.ndarray)
@expect.register(int)
@expect.register(float)
@apply_substitutions
def _expect_literal(self, expr=1, substitutions=None):
    if expr == 1:
        return self
    elif expr == 2:
        return jnp.square(self)
    elif expr == "log":
        return jnp.log(self)
    else:
        raise NotImplementedError


@expect.register
@apply_substitutions
def _expect_operator(self: Operator, expr: Any = 1, substitutions=None) -> jnp.ndarray:
    # Dispatch from an operator instance to the specific operation we're evaluating.
    return _expect_operator_unpacked(
        self.operation, expr, *self.args, **self.kwargs, substitutions=substitutions
    )


@valuedispatch
@reraise_not_implemented_with_args
def _expect_operator_unpacked(
    _, operator: Operator, expr: Any = 1, substitutions=None
) -> jnp.ndarray:
    raise NotImplementedError


@_expect_operator_unpacked.register(operator.matmul)
@reraise_not_implemented_with_args
@apply_substitutions
def _expect_operation_matmul(
    matmul, expr, *args, substitutions, **kwargs
) -> jnp.ndarray:
    sexpect = functools.partial(expect, substitutions=substitutions)
    # args = [substitutions[arg] for arg in operator.args]
    if expr == 1:
        args = [sexpect(arg) for arg in args]
        return functools.reduce(matmul, args[1:], args[0])
    elif expr == 2:
        # TODO: implement this more generally. We currently only support fixed design
        # matrix on the left and stochastic vector on the right. Batching isn't
        # supported.
        assert len(args) == 2
        design, coef = args
        assert isinstance(design, jnp.ndarray)
        assert isinstance(coef, jnp.ndarray) or (
            isinstance(coef, distributions.Distribution) and coef.event_dim == 0
        )
        return jnp.square(design) @ sexpect(coef, 2)
    else:
        raise NotImplementedError
