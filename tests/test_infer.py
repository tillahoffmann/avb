import avb
from avb.distributions import PrecisionNormal
import functools
import ifnt
import jax
from jax import numpy as jnp
import numpy as np
import numpyro
from numpyro import distributions


def linear_model(n=7, p=2, *, delay) -> None:
    if delay:
        new = avb.DelayedDistribution
    else:
        new = lambda cls, *args, **kwargs: cls(*args, **kwargs)  # noqa: E731

    X = numpyro.sample("X", new(PrecisionNormal, jnp.zeros((n, p)), 1))
    intercept = numpyro.sample("intercept", new(PrecisionNormal, 0, 1))
    coef = numpyro.sample(
        "coef", new(distributions.MultivariateNormal, jnp.zeros(p), jnp.eye(p))
    )
    tau = numpyro.sample("tau", new(distributions.Gamma, 4, 1))
    y_hat = intercept + X @ coef
    numpyro.sample("y", new(PrecisionNormal, y_hat, tau))


def test_delayed_eager_equivalence_and_structure() -> None:
    # First check that, in the eager mode, sampling is exactly as we'd expect from
    # numpyro.

    samples = []
    for delay in [False, True]:
        with numpyro.handlers.seed(rng_seed=17):
            trace = numpyro.handlers.trace(linear_model).get_trace(delay=delay)
        sample = {name: site["value"] for name, site in trace.items()}
        samples.append(sample)

    jax.tree.map(np.testing.assert_allclose, *samples)

    # Get a delayed trace and check its structure.
    with numpyro.handlers.trace() as delayed_trace, avb.delay():
        numpyro.handlers.trace(linear_model).get_trace(delay=True)

    assert set(delayed_trace) == {"X", "y", "coef", "intercept", "tau"}
    for name, site in delayed_trace.items():
        assert isinstance(site["value"], avb.nodes.DelayedValue), name
        assert isinstance(site["fn"], avb.nodes.DelayedDistribution), name


def test_expect_log_joint_linear_model() -> None:
    n = 7
    p = 2
    rng = ifnt.random.JaxRandomState(8)
    data = {
        "X": rng.normal((n, p)),
        "y": rng.normal((n,)),
    }
    conditioned = numpyro.handlers.condition(linear_model, data)
    approximation = {
        "tau": distributions.Gamma(*(rng.gamma(10, (2,)) / 10)),
        "coef": PrecisionNormal(rng.normal((p,)), rng.gamma(7, (p,)) / 7).to_event(),
        "intercept": PrecisionNormal(rng.normal(), rng.gamma(11) / 9),
    }
    elp = avb.expect_log_joint(conditioned, approximation, aggregate=False)(
        n, p, delay=True
    )

    # Verify numerically. This can probably be done more efficiently by batching. But
    # it's just a test.
    n_samples = 1000
    rng = ifnt.random.JaxRandomState(18)
    log_densities = {}
    for _ in range(n_samples):
        values = {
            key: (
                value.sample(rng.get_key())
                if isinstance(value, distributions.Distribution)
                else value
            )
            for key, value in approximation.items()
        }
        with numpyro.handlers.trace() as trace, numpyro.handlers.condition(data=values):
            conditioned(n, p, delay=False)

        for name, site in trace.items():
            log_densities.setdefault(name, []).append(
                site["fn"].log_prob(site["value"]).sum()
            )

    log_densities = {key: jnp.stack(value) for key, value in log_densities.items()}
    jax.tree.map(
        functools.partial(ifnt.testing.assert_samples_close, atol=1e-5),
        log_densities,
        elp,
    )


def test_elbo_linear_model() -> None:
    n = 7
    p = 2
    rng = ifnt.random.JaxRandomState(8)
    data = {
        "X": rng.normal((n, p)),
        "y": rng.normal((n,)),
    }
    conditioned = numpyro.handlers.condition(linear_model, data)
    approximation = {
        "tau": distributions.Gamma(*(rng.gamma(10, (2,)) / 10)),
        "coef": distributions.MultivariateNormal(
            rng.normal((p,)), jnp.eye(p) * rng.gamma(7, (p,)) / 7
        ),
        "intercept": PrecisionNormal(rng.normal(), rng.gamma(11) / 9),
    }
    elbo = avb.elbo_loss(conditioned, approximation)(n, p, delay=True)

    def guide(*args, **kwargs):
        return {
            name: numpyro.sample(name, dist) for name, dist in approximation.items()
        }

    trace_elbo = numpyro.infer.Trace_ELBO()
    n_samples = 1000
    trace_elbos = []
    for _ in range(n_samples):
        params = numpyro.handlers.seed(guide, rng.get_key())()
        trace_elbos.append(
            trace_elbo.loss(
                rng.get_key(), params, conditioned, guide, n, p, delay=False
            )
        )
    trace_elbos = jnp.stack(trace_elbos)

    ifnt.testing.assert_samples_close(trace_elbos, elbo)
