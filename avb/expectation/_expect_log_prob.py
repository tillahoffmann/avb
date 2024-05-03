import ifnt
from jax import numpy as jnp
from jax.scipy.special import gammaln, multigammaln
from numpyro import distributions
from ._expect import expect
from .. import dispatch
from ..distributions import LinearDynamicalSystem, PrecisionNormal
from ..nodes import DelayedValue
from ..util import as_distribution, tail_trace


@dispatch.classdispatch
def expect_log_prob(cls, value, *args, **kwargs) -> jnp.ndarray:
    raise NotImplementedError(cls)


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


@expect_log_prob.register(distributions.MultivariateNormal)
def _expect_log_prob_multivariate_normal(
    cls, value, loc, precision_matrix
) -> jnp.ndarray:
    # Take expectations. We add a trailing dimension to the first moments for easier
    # handling of matrix multiplications.
    precision_matrix1 = expect(precision_matrix)
    value, loc = DelayedValue.materialize(value, loc)
    value = as_distribution(value, event_dim=1)
    loc = as_distribution(loc, event_dim=1)
    value1 = expect(value)[..., None]
    valueo = expect(value, "outer")
    loc1 = expect(loc)[..., None]
    loco = expect(loc, "outer")

    # Evaluate the square in the exponent. We use the cyclical property of the trace to
    # evaluate the outer moments.
    squareform = (
        tail_trace(valueo @ precision_matrix1)
        - tail_trace(value1.mT @ precision_matrix1 @ loc1)
        - tail_trace(loc1.mT @ precision_matrix1 @ value1)
        + tail_trace(loco @ precision_matrix1)
    )

    p = precision_matrix1.shape[-1]
    return (
        expect(precision_matrix, "logabsdet") - p * jnp.log(2 * jnp.pi) - squareform
    ) / 2


@expect_log_prob.register(LinearDynamicalSystem)
def expect_log_prob_linear_dynamical_system(
    cls,
    value,
    transition_matrix: jnp.ndarray,
    innovation_precision,
) -> jnp.ndarray:
    r"""
    Evaluate the expected log probability for

    .. math::

        y_{t + 1} = A y_t + x_t

    for :math:`p`-dimensional state :math:`y\in\mathbb{R}^p` and innovations
    :math:`x\in\mathbb{R}^p`. We may equivalently write

    .. math::

        x_t = y_{t + 1} - A y_t

    such that the difference of adjacent states (after applying the appropriate
    transform) is equal to the innovation noise. The log probability is thus

    .. math::

        2 \log p(y) &= -pt \log\left(2\pi\right) + \log \left|\tau_0\right|
            + (t - 1)\log\left|\tau\right| \\
            &\qquad - y_1^\intercal \tau_0 y_1
            - \sum_{k=2} ^ t \left(y_k - A y_{k - 1}\right)^\intercal \tau
            \left(y_k - A y_{k - 1}\right).

    Args:
        value: State :code:`y`.
        transition_matrix: Transition matrix for the state.
        precision: Precision of innovations.
        init_precision: Precision of the first innovation (defaults to the usual
            precision).

    Returns:
        Expected log probability.
    """
    # Validate shapes.
    _, p = transition_matrix.shape
    outer = expect(value, "outer")
    assert outer.shape[-1] == p and outer.shape[-3] == p
    n_steps = outer.shape[-2]
    assert outer.shape[-2] == n_steps and outer.shape[-4] == n_steps
    batch_shape = outer.shape[:-4]

    # Reshape so we can use matrix multiplication in the state dimensions.
    outer = jnp.moveaxis(outer, -2, -3)
    assert outer.shape == batch_shape + (n_steps, n_steps, p, p)

    innovation_precision1 = expect(innovation_precision)
    innovation_logabsdet = expect(innovation_precision, "logabsdet")

    i = jnp.arange(n_steps)
    diag = ifnt.index_guard(outer)[..., i, i, :, :]
    diag_sum = diag.sum(axis=-3)
    i = jnp.arange(n_steps - 1)
    offdiag_sum = ifnt.index_guard(outer)[..., i, i + 1, :, :].sum(axis=-3)
    squareform = (
        # Contributions due to the first innovation with different precision.
        tail_trace(ifnt.index_guard(diag)[..., 0, :, :] @ innovation_precision1)
        # Contributions from remaining innovations.
        + tail_trace((diag_sum - diag[..., 0, :, :]) @ innovation_precision1)
        # Reactionary contributions.
        + tail_trace(
            (diag_sum - diag[..., -1, :, :])
            @ transition_matrix.T
            @ innovation_precision1
            @ transition_matrix
        )
        # Interactions.
        - 2 * tail_trace(offdiag_sum @ innovation_precision1 @ transition_matrix)
    )

    result = (
        n_steps * innovation_logabsdet - squareform - n_steps * p * jnp.log(2 * jnp.pi)
    ) / 2
    ifnt.testing.assert_shape(result, batch_shape)
    return result
