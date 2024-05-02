import avb
from avb.distributions import PrecisionNormal
import functools
import ifnt
import jax
from jax import numpy as jnp
import numpy as np
import numpyro
from numpyro import distributions


def model(delay, n, p) -> None:
    if delay:
        new = avb.DelayedDistribution
    else:
        new = lambda cls, *args, **kwargs: cls(*args, **kwargs)  # noqa: E731

    X = numpyro.sample("X", new(PrecisionNormal, jnp.zeros((n, p)), 1))
    intercept = numpyro.sample("intercept", new(PrecisionNormal, 0, 1))
    coef = numpyro.sample("coef", new(PrecisionNormal, jnp.zeros(p), 1))
    tau = numpyro.sample("tau", new(distributions.Gamma, 4, 1))
    y_hat = intercept + X @ coef
    numpyro.sample("y", new(PrecisionNormal, y_hat, tau))


def test_smoke() -> None:
    # First check that, in the eager mode, sampling is exactly as we'd expect from
    # numpyro.
    n = 50
    p = 2

    samples = []
    for delay in [False, True]:
        with numpyro.handlers.seed(rng_seed=17):
            trace = numpyro.handlers.trace(model).get_trace(delay, n, p)
        sample = {name: site["value"] for name, site in trace.items()}
        samples.append(sample)

    jax.tree.map(np.testing.assert_allclose, *samples)

    # Validate the structure of the delayed trace.
    with numpyro.handlers.trace() as delayed_trace, avb.delay():
        numpyro.handlers.trace(model).get_trace(True, n, p)

    assert set(trace) == set(delayed_trace)
    for name, site in delayed_trace.items():
        assert isinstance(site["value"], avb.nodes.DelayedValue), name
        assert isinstance(site["fn"], avb.nodes.DelayedDistribution), name

    # Create the substitution for which we want to evaluate the expected log joint
    # distribution. Substitutions are either concrete values or variational factors to
    # be optimized.
    substitutions = sample | {
        "tau": distributions.Gamma(4, 3),
        "coef": PrecisionNormal(jnp.asarray([-0.2, 0.3]), jnp.asarray([0.9, 1.1])),
        "intercept": PrecisionNormal(0.4, 1.4),
    }
    with (
        numpyro.handlers.trace() as conditioned_trace,
        avb.delay(),
        numpyro.handlers.condition(data=substitutions),
    ):
        numpyro.handlers.trace(model).get_trace(True, n, p)

    # Evaluate the expected log joitn distribution by iterating over all sites.
    elp = {}
    for name, site in conditioned_trace.items():
        fn: avb.DelayedDistribution = site["fn"]
        value = avb.expect_log_prob(
            fn.cls,
            site["value"],
            *fn.args,
            **fn.kwargs,
        ).sum()
        assert jnp.isfinite(value).all()
        elp[name] = value

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
            for key, value in substitutions.items()
        }
        with numpyro.handlers.trace() as trace, numpyro.handlers.condition(data=values):
            model(False, n, p)

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
