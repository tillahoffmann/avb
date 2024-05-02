from jax import numpy as jnp
from jax.scipy.special import gammaln, multigammaln
from numpyro import distributions
from ._expect import expect
from .. import dispatch
from ..util import tail_trace


@dispatch.classdispatch
def expect_log_prob(cls, value, *args, **kwargs) -> jnp.ndarray:
    raise NotImplementedError


@expect_log_prob.register(PrecisionNormal)
def _expect_log_prob_precision_normal(cls, value, loc, precision) -> jnp.ndarray:
    return (
        expect(precision, "log")
        - jnp.log(2 * jnp.pi)
        - expect(precision)
        * (expect(loc, 2) - 2 * expect(loc) * expect(value) + expect(value, 2))
    ) / 2


@expect_log_prob.register(distributions.Gamma)
def _expect_log_prob_gamma(cls, value, concentration, rate) -> jnp.ndarray:
    normalize_term = gammaln(concentration) - concentration * expect(rate, "log")
    return (
        (concentration - 1) * expect(value, "log")
        - expect(rate) * expect(value)
        - normalize_term
    )


@expect_log_prob.register(distributions.Wishart)
def _expect_log_prob_wishart(cls, value, concentration, rate_matrix) -> jnp.ndarray:
    value1 = expect(value)
    p = value1.shape[-1]
    return (
        (concentration - p - 1) * expect(value, "logabsdet") / 2
        - tail_trace(expect(rate_matrix) @ value1) / 2
        - concentration * p / 2 * jnp.log(2)
        + concentration / 2 * expect(rate_matrix, "logabsdet")
        - multigammaln(concentration / 2, p)
    )

