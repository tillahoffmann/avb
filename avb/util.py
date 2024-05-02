from jax import numpy as jnp
from jax.scipy import special


def tail_trace(a: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluate trace along the last two dimensions.
    """
    # https://github.com/google/jax/pull/20970
    return jnp.diagonal(a, axis1=-1, axis2=-2).sum(axis=-1)


def multidigamma(a: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
    """
    Returns the derivative of the log of multivariate gamma.
    """
    # Based on https://bayespy.org/_modules/bayespy/utils/misc.html#multidigamma.
    return special.digamma(a[..., None] - 0.5 * jnp.arange(d)).sum(axis=-1)
