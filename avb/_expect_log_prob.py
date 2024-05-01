import functools
from jax import numpy as jnp
from jax.scipy.special import gammaln
from numpyro import distributions
from ._expect import apply_substitutions, expect
from . import dispatch
from .distributions import PrecisionNormal


@dispatch.classdispatch
@dispatch.reraise_not_implemented_with_args
def expect_log_prob(cls, value, *args, **kwargs):
    raise NotImplementedError


@expect_log_prob.register(PrecisionNormal)
@apply_substitutions
def _expect_log_prob_precision_normal(
    cls, value, loc, precision, *, substitutions=None
) -> jnp.ndarray:
    sexpect = functools.partial(expect, substitutions=substitutions)
    return (
        sexpect(precision, "log")
        - jnp.log(2 * jnp.pi)
        - sexpect(precision)
        * (sexpect(loc, 2) - 2 * sexpect(loc) * sexpect(value) + sexpect(value, 2))
    ) / 2


@expect_log_prob.register(distributions.Gamma)
@apply_substitutions
def _expect_log_prob_gamma(
    cls, value, concentration, rate, *, substitutions=None
) -> jnp.ndarray:
    normalize_term = gammaln(concentration) - concentration * expect(rate, "log")
    sexpect = functools.partial(expect, substitutions=substitutions)
    return (
        (concentration - 1) * sexpect(value, "log")
        - sexpect(rate) * sexpect(value)
        - normalize_term
    )
