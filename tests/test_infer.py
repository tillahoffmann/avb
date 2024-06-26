import avb
from avb.distributions import PoissonLogits, PrecisionNormal
import functools
import ifnt
import jax
from jax import numpy as jnp
import jaxopt
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
    # Observation-level random effects for overdispersion.
    olre = numpyro.sample("olre", new(PrecisionNormal, jnp.zeros(n), tau))
    y_hat = intercept + olre + X @ coef
    numpyro.sample("y", new(PoissonLogits, y_hat))


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

    assert set(delayed_trace) == {"X", "y", "coef", "intercept", "tau", "olre"}
    for name, site in delayed_trace.items():
        assert isinstance(site["value"], avb.nodes.DelayedValue), name
        assert isinstance(site["fn"], avb.nodes.DelayedDistribution), name
        assert not site["value"].has_value
        assert site["value"].shape == trace[name]["value"].shape


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
        "olre": PrecisionNormal(rng.normal((n,)), rng.gamma(100, (n,))),
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
    for key, value in log_densities.items():
        ifnt.testing.assert_samples_close(value, elp[key].sum(), atol=1e-5)


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
        "olre": PrecisionNormal(rng.normal((n,)), rng.gamma(100, (n,))),
    }

    avb.infer.validate_elbo(conditioned, approximation)(rng.get_key(), n, p, delay=True)


def test_end_to_end_linear_model() -> None:
    n = 500
    p = 5
    with numpyro.handlers.trace() as trace, numpyro.handlers.seed(rng_seed=9):
        linear_model(n, p, delay=False)
    data = {"X": trace["X"]["value"], "y": trace["y"]["value"]}
    # We use a partial model because jit-compiling the "delay" parameter causes
    # TracerBoolConversionError.
    partial_model = functools.partial(linear_model, n, p, delay=True)
    conditioned = numpyro.handlers.condition(partial_model, data)
    approximation = {
        "tau": distributions.Gamma(jnp.ones(()), jnp.ones(())),
        "coef": distributions.LowRankMultivariateNormal(
            jnp.zeros(p), jnp.zeros((p, 2)), 0.1 * jnp.ones(p)
        ),
        "intercept": PrecisionNormal(jnp.zeros(()), 0.1 * jnp.ones(())),
        "olre": PrecisionNormal(jnp.zeros(n), 0.1 * jnp.ones(n)),
    }
    unconstrained, aux = avb.approximation_to_unconstrained(approximation)
    # This wrapper contains all the static information so we can jit-compile it.
    loss_fn = jax.jit(avb.elbo_loss_from_unconstrained(conditioned, aux))
    solver = jaxopt.LBFGS(loss_fn, maxiter=1000)
    result = solver.run(unconstrained)
    approximation = avb.approximation_from_unconstrained(result.params, aux)
    assert jnp.corrcoef(approximation["coef"].mean, trace["coef"]["value"])[0, 1] > 0.99
    assert jnp.corrcoef(approximation["olre"].mean, trace["olre"]["value"])[0, 1] > 0.5
