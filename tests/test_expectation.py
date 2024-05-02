import avb
import ifnt
from jax import numpy as jnp
from numpyro import distributions
import operator
import pytest
from typing import Any, Type


rng = ifnt.random.JaxRandomState(9)


@pytest.mark.parametrize(
    "distribution, expr",
    [
        (distributions.Gamma(3.4, 5.7), 1),
        (distributions.Gamma(3.4, 5.7), 2),
        (distributions.Gamma(3.4, 5.7), "log"),
        (avb.distributions.PrecisionNormal(1.2, 4.1), 1),
        (avb.distributions.PrecisionNormal(1.2, 4.1), 2),
        (
            avb.nodes.Operator(
                operator.add,
                avb.distributions.PrecisionNormal(1.2, 4.1),
                distributions.Gamma(3.4, 5.7),
            ),
            1,
        ),
        (
            avb.nodes.Operator(
                operator.add,
                avb.distributions.PrecisionNormal(1.2, 4.1),
                distributions.Gamma(3.4, 5.7),
            ),
            2,
        ),
        (
            avb.nodes.Operator(
                operator.getitem,
                avb.distributions.PrecisionNormal(
                    rng.normal((11,)), rng.gamma(10, (11,)) / 10
                ),
                (Ellipsis, 2 * jnp.arange(5)),
            ),
            2,
        ),
        (
            avb.nodes.Operator(
                operator.getitem,
                distributions.Gamma(rng.gamma(10, (11,)) / 10),
                (Ellipsis, 2 * jnp.arange(5)),
            ),
            "log",
        ),
    ],
)
def test_expect(distribution: distributions.Distribution, expr: Any) -> None:
    expr2func = {
        1: lambda x: x,
        2: jnp.square,
        "log": jnp.log,
    }
    rng = ifnt.random.JaxRandomState(23)
    n_samples = 1000
    if isinstance(distribution, avb.nodes.Operator):
        x = distribution.operator(
            *(
                (
                    arg.sample(rng.get_key(), (n_samples,))
                    if isinstance(arg, distributions.Distribution)
                    else arg
                )
                for arg in distribution.args
            )
        )
    else:
        x = distribution.sample(rng.get_key(), (n_samples,))
    ifnt.testing.assert_samples_close(
        expr2func[expr](x), avb.expect(distribution, expr)
    )


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
    ],
)
def test_expect_log_prob(cls: Type[distributions.Distribution], params: dict) -> None:
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
