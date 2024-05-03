import avb
from avb.nodes import Operator
import functools
import ifnt
import jax
from jax import numpy as jnp
from numpyro import distributions
import operator as builtin_operator
import pytest
from typing import Any, Type


rng = ifnt.random.JaxRandomState(9)


# Pairs of distributions and tuple of the statistics to check.
DISTRIBUTION_CONFIGS = [
    (distributions.Gamma(3.4, 5.7), (1, 2, "log", "var")),
    (avb.distributions.PrecisionNormal(1.2, 4.1), (1, 2, "var")),
    (
        avb.distributions.Reshaped(
            distributions.Normal(rng.normal((5, 6, 4, 3)), 0.1).to_event(2),
            batch_shape=(3, 10),
            event_shape=(12,),
        ),
        (1, 2, "outer", "var"),
    ),
    (
        distributions.MultivariateNormal(
            rng.normal((3,)),
            distributions.Wishart(6, jnp.eye(3) / 10).sample(rng.get_key()),
        ),
        (1, 2, "outer", "var"),
    ),
    (
        distributions.Wishart(
            4, distributions.Wishart(4, jnp.eye(3) / 10).sample(rng.get_key())
        ),
        (1, 2, "logabsdet", "var"),
    ),
]


@pytest.mark.parametrize(
    "distribution, expr",
    [(dist, expr) for dist, exprs in DISTRIBUTION_CONFIGS for expr in exprs],
)
def test_expect_distribution(
    distribution: distributions.Distribution, expr: Any
) -> None:
    rng = ifnt.random.JaxRandomState(23)
    n_samples = 10000
    x = distribution.sample(rng.get_key(), (n_samples,))
    if expr == "var":
        x = x.reshape((100, 100) + x.shape[1:]).var(axis=0)
    else:
        x = distributions.Delta(x, event_dim=distribution.event_dim)
        x = avb.expect(x, expr)
    ifnt.testing.assert_samples_close(x, avb.expect(distribution, expr))


# Pairs of operators whose arguments are distributions and tuple of the statistics to
# check.
OPERATOR_CONFIGS = [
    (
        Operator(
            builtin_operator.add,
            distributions.Normal(1.4, 0.2),
            distributions.Normal(3.4, 0.3),
        ),
        (1, 2, "var"),
    ),
    (
        Operator(
            builtin_operator.matmul,
            distributions.Normal(1.4 * jnp.ones((3, 2)), 0.2).to_event(1),
            distributions.Normal(3.4 * jnp.ones(2), 0.3).to_event(),
        ),
        (1, 2, "var"),
    ),
    (
        Operator(
            builtin_operator.getitem,
            distributions.Gamma(rng.gamma(10, (5, 2)), 0.2),
            (Ellipsis, jnp.asarray([2, 4]), slice(None)),
        ),
        (1, 2, "var", "log"),
    ),
]


@pytest.mark.parametrize(
    "operator, expr",
    [(operator, expr) for operator, exprs in OPERATOR_CONFIGS for expr in exprs],
)
def test_expect_operator(operator: Operator, expr: Any) -> None:
    def _sample_operator(key, operator: Operator) -> jnp.ndarray:
        args = []
        for arg in operator.args:
            key, subkey = jax.random.split(key)
            if isinstance(arg, distributions.Distribution):
                arg = arg.sample(subkey)
            elif isinstance(arg, Operator):
                arg = _sample_operator(subkey, arg)
            args.append(arg)
        return operator.operator(*args)

    # We vmap here rather than batch because matmul behaves differently depending on
    # the exact shapes.
    n_samples = 10000
    keys = jax.random.split(jax.random.key(27), n_samples)
    x = jax.vmap(functools.partial(_sample_operator, operator=operator))(keys)
    if expr == "var":
        x = x.reshape((100, 100) + x.shape[1:]).var(axis=0)
    else:
        x = avb.expect(x, expr)
    ifnt.testing.assert_samples_close(x, avb.expect(operator, expr))


@pytest.mark.parametrize(
    "cls, params",
    [
        (
            distributions.Gamma,
            {"concentration": 4.2, "rate": distributions.Gamma(10, 5)},
        ),
        (
            avb.distributions.PrecisionNormal,
            {
                "loc": distributions.Normal(0.5, 1.2),
                "precision": distributions.Gamma(7.5, 9),
            },
        ),
        (
            distributions.Wishart,
            {"concentration": 7, "rate_matrix": distributions.Wishart(20, jnp.eye(3))},
        ),
        (
            distributions.MultivariateNormal,
            {
                "loc": distributions.Normal().expand((3,)).to_event(),
                "precision_matrix": distributions.Wishart(20, jnp.eye(3)),
            },
        ),
    ],
)
def test_expect_log_prob(cls: Type[distributions.Distribution], params: dict) -> None:
    # Create an instance of the distribution which will later serve as the "value".
    rng = ifnt.random.JaxRandomState(75)
    instance_params = {
        key: (
            value.sample(rng.get_key())
            if isinstance(value, distributions.Distribution)
            else value
        )
        for key, value in params.items()
    }
    dist = cls(**instance_params)

    # Sanity check that the expected log probability is correct for a point mass.
    x = dist.sample(rng.get_key())
    ifnt.testing.assert_allclose(
        avb.expect_log_prob(cls, x, **instance_params), dist.log_prob(x), rtol=1e-5
    )

    # Sample from the parameter distributions to construct an ensemble of parameters to
    # average over. Then compare with the expected log joint.
    n_samples = 1000
    expect_params = {
        key: (
            value.sample(rng.get_key(), (n_samples,))
            if isinstance(value, distributions.Distribution)
            else value
        )
        for key, value in params.items()
    }
    values = dist.sample(rng.get_key(), (n_samples,))
    log_probs = cls(**expect_params).log_prob(values)

    ifnt.testing.assert_samples_close(
        log_probs, avb.expect_log_prob(cls, dist, **params)
    )
