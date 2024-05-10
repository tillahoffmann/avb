import functools
import ifnt
import jax
from jax import numpy as jnp
import numpyro
from numpyro import distributions, handlers
from typing import Callable, Dict, Union
from .expectation import expect_log_prob
from .nodes import delay, DelayedDistribution, DelayedValue, materialize
from .unconstrained import approximation_from_unconstrained


def expect_log_joint(
    model: Callable, approximation: Dict, aggregate: bool = True
) -> Callable[..., Union[jnp.ndarray, Dict[str, jnp.ndarray]]]:
    """
    Transform a model to evaluate the expected log joint density under a posterior
    approximation.

    Args:
        model: Model to transform.
        approximation: Posterior approximation for each site.
        aggregate: Aggregate results to return a single scalar.

    Returns:
        Callable returning a dictionary of expected log densities per site or expected
        log joint density as a scalar if :code:`aggregate` is true.
    """

    @functools.wraps(model)
    def _expect_log_joint_wrapper(*args, **kwargs):
        with handlers.trace() as trace, delay(), handlers.condition(data=approximation):
            model(*args, **kwargs)

        result = {}
        for name, site in trace.items():
            fn = site["fn"]
            value = site["value"]
            name = site["name"]
            assert isinstance(fn, DelayedDistribution), (
                f"Distribution for site '{name}' must be an instance of "
                f"`DelayedDistribution` but got {fn}."
            )
            assert isinstance(value, DelayedValue), (
                f"Value for site '{name}' must be an instance of `DelayedValue` but "
                f"got {value}."
            )
            assert value.has_value, f"Site '{name}' must have a value."
            elp = expect_log_prob(
                fn.cls,
                value,
                *fn.args,
                **fn.kwargs,
            )
            ifnt.testing.assert_allfinite(elp)
            result[name] = elp

        if aggregate:
            return sum(x.sum() for x in result.values())
        return result

    return _expect_log_joint_wrapper


def elbo_loss(model: Callable, approximation: Dict) -> Callable[..., jnp.ndarray]:
    """
    Transform a model to evaluate the *negative* evidence lower bound under a posterior
    approximation.

    Args:
        model: Model to transform.
        approximation: Posterior approximation for each site.

    Returns:
        Callable evaluating the evidence lower bound.
    """
    # We can precompute the entropy and transform the model to get the log joint.
    entropy = sum(factor.entropy().sum() for factor in approximation.values())
    log_joint_fn = expect_log_joint(model, approximation)

    @functools.wraps(model)
    def _elbo_wrapper(*args, **kwargs) -> jnp.ndarray:
        return -log_joint_fn(*args, **kwargs) - entropy

    return _elbo_wrapper


def elbo_loss_from_unconstrained(
    model: Callable, aux: Dict, *, validate_args=None
) -> Callable[..., jnp.ndarray]:
    """
    Transform a model to evaluate the *negative* evidence lower bound under a posterior
    approximation defined by unconstrained, optimizable parameters.

    Args:
        model: Model to transform.
        aux: Static auxiliary information keyed by factor name.
        validate_args: Validate distribution arguments.

    Returns:
        Callable evaluating the evidence lower bound given unconstrained parameters.
    """

    def _elbo_loss_from_unconstrained(unconstrained: Dict, *args, **kwargs):
        approximation = approximation_from_unconstrained(
            unconstrained, aux, validate_args=validate_args
        )
        return elbo_loss(model, approximation)(*args, **kwargs)

    return _elbo_loss_from_unconstrained


def guide_from_approximation(
    approximation: Dict[str, distributions.Distribution]
) -> Callable:
    """
    Create a numpyro guide from a factorized approximation.

    Args:
        approximation: Mapping of site names to variational factors.

    Returns:
        Callable guide.
    """

    def _guide_from_approximation_wrapper(*args, **kwargs) -> None:
        for name, dist in approximation.items():
            numpyro.sample(name, dist)

    return _guide_from_approximation_wrapper


def validate_elbo(
    model: Callable,
    approximation: Dict[str, distributions.Distribution],
    n_samples: int = 1000,
    verbose: bool = True,
    **samples_close_kwargs,
) -> Callable:
    """
    Validate evidence lower bound of the model under the variational approximation by
    comparing with a Monte Carlo estimate using numpyro.

    Args:
        model: Model to validate.
        approximation: Variational factors to validate.
        n_samples: Number of Monte Carlo samples.
        verbose: Print information about the validation.
        **samples_close_kwargs: Keyword arguments passed to
            :func:`ifnt.testing.assert_samples_close`.
    """

    def _validate_elbo_wrapper(key, *args, **kwargs):
        # Evaluate the elbo analytically.
        expected = elbo_loss(model, approximation)(*args, **kwargs)

        # Create a guide from the factorized approximation and materialize the model
        # so there are only concrete numpyro.distributions.Distribution rather than
        # DelayedDistribution wrappers.
        guide = guide_from_approximation(approximation)
        materialized_model = materialize(model)

        # Scan to get elbo samples.
        def _body(key, _):
            key, guide_key, model_key = jax.random.split(key, 3)
            seeded_guide = numpyro.handlers.seed(guide, guide_key)
            with handlers.trace() as trace:
                seeded_guide(*args, **kwargs)
            params = {name: site["value"] for name, site in trace.items()}
            trace_elbo = numpyro.infer.Trace_ELBO()
            loss = trace_elbo.loss(
                model_key, params, materialized_model, guide, *args, **kwargs
            )
            return key, loss

        _, elbos = jax.lax.scan(_body, key, jnp.arange(n_samples))

        ifnt.testing.assert_samples_close(elbos, expected, **samples_close_kwargs)

        if verbose:
            mean = elbos.mean()
            stderr = elbos.std() / jnp.sqrt(n_samples - 1)
            z = (expected - mean) / stderr

            print(
                f"Sample mean {mean} with standard error {stderr} is consistent with "
                f"the analytical value {expected} (z-score = {z})."
            )

        return elbos, expected

    return _validate_elbo_wrapper
