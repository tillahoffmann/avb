import avb
import ifnt
import jax
from jax import numpy as jnp
from numpyro import distributions
import pytest


rng = ifnt.random.JaxRandomState(43)


@pytest.mark.parametrize(
    "dist",
    [
        distributions.Gamma(rng.gamma(5, (13, 4)), rng.gamma(7, (4,))),
        avb.distributions.PrecisionNormal(rng.normal((7,)), rng.gamma(7, (7,))),
        distributions.LowRankMultivariateNormal(
            rng.normal((7, 4)), rng.normal((7, 4, 5)), rng.gamma(5, (7, 4))
        ),
        avb.distributions.Reshaped(
            distributions.Gamma(
                rng.gamma(5, (13, 7, 4, 3)), rng.gamma(7, (7, 4, 3))
            ).to_event(2),
            batch_shape=(91,),
            event_shape=(6, 2),
        ),
        distributions.Wishart(9, jnp.eye(3)),
    ],
)
def test_unconstrained_round_trip(dist: distributions.Distribution) -> None:
    # Run one round trip and verify the log probs are the same.
    unconstrained, aux = avb.to_unconstrained(dist)
    other = avb.from_unconstrained(unconstrained, aux)
    x = dist.sample(rng.get_key())
    ifnt.testing.assert_allclose(other.log_prob(x), dist.log_prob(x), atol=1e-5)

    # Randomize the unconstrained parameters and construct a distribution, validating
    # arguments. The distribution should still be valid unless the arg constraints are
    # wrong. We need to call `to_unconstrained` again because `from_unconstrained`
    # modifies its arguments in place.
    unconstrained, aux = avb.to_unconstrained(dist)
    unconstrained = jax.tree.map(lambda x: rng.normal(x.shape), unconstrained)
    other = avb.from_unconstrained(unconstrained, aux, validate_args=True)
    log_prob = other.log_prob(x)
    ifnt.testing.assert_allfinite(log_prob)


def test_unconstrained_approximation_round_trip() -> None:
    approximation = {
        "tau": distributions.Gamma(*(rng.gamma(10, (2,)) / 10)),
        "coef": avb.distributions.PrecisionNormal(
            rng.normal((3,)), rng.gamma(7, (3,)) / 7
        ).to_event(),
        "intercept": avb.distributions.PrecisionNormal(rng.normal(), rng.gamma(11) / 9),
    }
    unconstrained, aux = avb.approximation_to_unconstrained(approximation)
    other = avb.approximation_from_unconstrained(unconstrained, aux)

    for key, factor in approximation.items():
        x = factor.sample(rng.get_key())
        ifnt.testing.assert_allclose(factor.log_prob(x), other[key].log_prob(x))
