import functools
import ifnt
from jax import numpy as jnp
from numpyro import handlers
from typing import Callable, Dict, Union
from .expectation import expect_log_prob
from .nodes import delay, DelayedDistribution


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
            assert isinstance(fn, DelayedDistribution)
            # FIXME: Do not sum blindly over all dimensions but make sure we have
            # log_prob shape equal to the batch shape of the distribution.
            value = expect_log_prob(
                fn.cls,
                site["value"],
                *fn.args,
                **fn.kwargs,
            ).sum()
            ifnt.testing.assert_allfinite(value)
            result[name] = value

        if aggregate:
            return sum(result.values())
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
