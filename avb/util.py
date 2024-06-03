import jax
from jax import numpy as jnp
from jax.scipy import special
import numbers
from numpyro import distributions


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


def as_distribution(value, event_dim=None):
    if isinstance(value, distributions.Distribution):
        if event_dim is not None and event_dim != value.event_dim:
            raise ValueError(
                f"Expected {event_dim} event dimensions; got {value.event_dim}."
            )
        return value
    return distributions.Delta(value, event_dim=event_dim)


def get_shape(x) -> tuple:
    if isinstance(x, numbers.Number):
        return ()
    elif isinstance(x, distributions.Distribution):
        return x.shape()
    else:
        return x.shape


def tree_leaves_with_path(tree, is_leaf=None, sep=None) -> list:
    leaves = jax.tree_util.tree_leaves_with_path(tree, is_leaf)
    if not sep:
        return leaves
    return [("/".join(key.key for key in keys), leaf) for keys, leaf in leaves]


def apply_scale(x, scales, strict: bool = False):
    """
    Scale all values in `x` by factors in `scales` if a corresponding element exists. If
    `strict`, the pytrees of `x` and `scales` must match exactly.
    """
    if strict:
        return jax.tree.map(lambda x, y: x * y, x, scales)
    else:
        # Flatten the scales so we can look them up.
        flat_scales, _ = jax.tree_util.tree_flatten_with_path(scales)
        flat_scales = dict(flat_scales)
        return jax.tree_util.tree_map_with_path(
            lambda key, value: (
                (flat_scales[key] * value) if key in flat_scales else value
            ),
            x,
        )


def precondition_diagonal(func, scales, strict: bool = False):
    """
    Precondition a function by scaling its first argument.
    """

    def _precondition_diagonal_wrapper(x, *args, **kwargs):
        x = apply_scale(x, scales, strict=strict)
        return func(x, *args, **kwargs)

    return _precondition_diagonal_wrapper


def hessian_diagonal_finite_diff(func, x, *args, eps=1e-6, **kwargs):
    """
    Compute the diagonal Hessian using symmetric finite difference of the autodiff grad.
    """

    grad = jax.grad(func)
    grad1 = grad(jax.tree.map(lambda x: x - eps, x), *args, **kwargs)
    grad2 = grad(jax.tree.map(lambda x: x + eps, x), *args, **kwargs)
    return jax.tree.map(lambda x, y: (y - x) / (2 * eps), grad1, grad2)
