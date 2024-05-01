import avb
from avb.distributions import PrecisionNormal
import ifnt
import jax
from jax import numpy as jnp
import numpy as np
import numpyro
from numpyro import distributions
from numpyro.infer.util import log_density


def model(use_lazy_distribution, n, p) -> None:
    if use_lazy_distribution:
        new = avb.LazyDistribution
    else:
        new = lambda cls, *args, **kwargs: cls(*args, **kwargs)  # noqa: E731

    X = numpyro.sample("X", new(PrecisionNormal, jnp.zeros((n, p)), 1))
    coef = numpyro.sample("coef", new(PrecisionNormal, jnp.zeros(p), 1))
    tau = numpyro.sample("tau", new(distributions.Gamma, 4, 1))
    y_hat = X @ coef
    numpyro.sample("y", new(PrecisionNormal, y_hat, tau))


def test_smoke() -> None:
    # First check that, in the eager mode, sampling is exactly as we'd expect from
    # numpyro.
    n = 50
    p = 2

    samples = []
    for use_lazy_distribution in [False, True]:
        with numpyro.handlers.seed(rng_seed=17):
            trace = numpyro.handlers.trace(model).get_trace(use_lazy_distribution, n, p)
        sample = {name: site["value"] for name, site in trace.items()}
        samples.append(sample)

    jax.tree.map(np.testing.assert_allclose, *samples)

    # Validate the structure of the lazy trace.
    with avb.LazyDistribution.lazy_trace():
        lazy_trace = numpyro.handlers.trace(model).get_trace(True, n, p)

    assert set(trace) == set(lazy_trace)
    for name, site in lazy_trace.items():
        assert isinstance(site["value"], avb.nodes.LazySample), name
        assert isinstance(site["fn"], avb.nodes.LazyDistribution), name

    # Create the substitution for which we want to evaluate the expected log joint
    # distribution. Substitutions are either concrete values or variational factors to
    # be optimized. We first create them by name and then use those names to map to the
    # lazy sample instances.
    approximation = sample | {
        "tau": distributions.Gamma(4, 3),
        "coef": PrecisionNormal(jnp.asarray([-0.2, 0.3]), jnp.asarray([0.9, 1.1])),
    }
    substitutions = {
        lazy_trace[key]["value"]: value for key, value in approximation.items()
    }

    # Evaluate the expected log joint distribution by iterating over all sites.
    elp = {}
    for name, site in lazy_trace.items():
        fn: avb.LazyDistribution = site["fn"]
        value = avb.expect_log_prob(
            fn.cls,
            substitutions[site["value"]],
            *fn.args,
            **fn.kwargs,
            substitutions=substitutions,
        )
        assert jnp.isfinite(value).all()
        elp[name] = value

    # Verify numerically. This can probably be done more efficiently by batching.
    n_samples = 1000
    rng = ifnt.random.JaxRandomState(18)
    log_densities = []
    for _ in range(n_samples):
        values = {
            key: (
                value.sample(rng.get_key())
                if isinstance(value, distributions.Distribution)
                else value
            )
            for key, value in approximation.items()
        }
        ld, _ = log_density(model, (False, n, p), {}, values)
        log_densities.append(ld)
    log_densities = jnp.stack(log_densities)

    # Hacky way to add up the log density contributions.
    actual = sum(jax.tree.map(jnp.sum, elp).values())
    ifnt.testing.assert_samples_close(log_densities, actual)
