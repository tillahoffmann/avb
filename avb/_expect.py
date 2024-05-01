import functools
from jax import numpy as jnp
from jax.scipy.special import digamma
from numpyro import distributions
from typing import Any
from .dispatch import reraise_not_implemented_with_args


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
