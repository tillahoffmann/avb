import avb
import ifnt
from jax import numpy as jnp
from numpyro import distributions


def test_linear_dynamical_system_log_prob() -> None:
    n_steps = 5
    p = 3
    batch_shape = (11, 17)

    rng = ifnt.random.JaxRandomState(9)
    transition_matrix = rng.normal((p, p))
    innovation_precision = distributions.Wishart(2 * p, jnp.eye(p)).sample(
        rng.get_key(), batch_shape
    )
    dist = avb.distributions.LinearDynamicalSystem(
        transition_matrix, innovation_precision, n_steps=n_steps
    )
    assert dist.shape() == batch_shape + (n_steps, p)
    x = dist.sample(rng.get_key())
    log_prob = dist.log_prob(x)

    # Compare with evaluation of the log probability using a full-rank multivariate
    # normal distribution.
    covariance_matrix = dist.covariance_tensor.reshape(
        batch_shape + (n_steps * p, n_steps * p)
    )
    other_dist = distributions.MultivariateNormal(covariance_matrix=covariance_matrix)
    other_log_prob = other_dist.log_prob(x.reshape(batch_shape + (n_steps * p,)))
    ifnt.testing.assert_allclose(log_prob, other_log_prob, rtol=1e-4)

    # For a transition matrix of zeros, the samples should be independent, and the
    # log prob should be equivalent to independent samples.
    transition_matrix = jnp.zeros((p, p))
    dist = avb.distributions.LinearDynamicalSystem(
        transition_matrix, innovation_precision, n_steps=n_steps
    )
    log_prob = dist.log_prob(x)
    other_dist = distributions.MultivariateNormal(
        0, precision_matrix=innovation_precision[..., None, :, :]
    ).to_event(1)
    other_log_prob = other_dist.log_prob(x)
    ifnt.testing.assert_allclose(log_prob, other_log_prob, rtol=1e-5)
