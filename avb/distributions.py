import ifnt
import jax
from jax import numpy as jnp
import math
from numpyro import distributions
from numpyro.distributions.util import lazy_property
from typing import Optional


class PrecisionNormal(distributions.Normal):
    arg_constraints = {
        "loc": distributions.constraints.real,
        "precision": distributions.constraints.positive,
    }

    def __init__(self, loc=0.0, precision=1.0, *, validate_args=None) -> None:
        super().__init__(
            loc=loc, scale=1.0 / jnp.sqrt(precision), validate_args=validate_args
        )


def _evaluate_markov_precision(
    *,
    transition_matrix: jnp.ndarray,
    innovation_precision: jnp.ndarray,
    init_precision: jnp.ndarray,
    n_steps: int,
) -> jnp.ndarray:
    """
    Evaluate the state space precision.

    Args:
        transition_matrix: Transition matrix with shape `(..., p, p)`, where `p` is the
            dimensionality of the state space.
        innovation_precision: Precision of innovation noise with shape `(..., p, p)`.
        init_precision: Precision of the initial state with shape `(..., p, p)`.
        n_steps: Number of steps.

    Returns:
        Precision of the state with shape `(..., n * p, n * p)`.
    """
    # Broadcast arrays and reshape to have a single batch dimension. We'll restore it
    # later.
    transition_matrix, innovation_precision, init_precision = jnp.broadcast_arrays(
        transition_matrix, innovation_precision, init_precision
    )
    *batch_shape, p, _ = transition_matrix.shape
    transition_matrix = transition_matrix.reshape((-1, p, p))
    innovation_precision = innovation_precision.reshape((-1, p, p))
    init_precision = init_precision.reshape((-1, p, p))

    # Evaluate one of the blocks of the precision matrix which we'll pad and roll.
    negative_offdiag_block = transition_matrix.mT @ innovation_precision
    diag_block = (negative_offdiag_block @ transition_matrix) + innovation_precision
    row_block = jnp.concatenate(
        [-negative_offdiag_block.mT, diag_block, -negative_offdiag_block], axis=-1
    )

    # Pad for rolling, vmap the roll for efficiency, and discard the part we don't need.
    result = jnp.pad(row_block, ((0, 0), (0, 0), (0, (n_steps - 1) * p)))
    result = jax.vmap(lambda shift: jnp.roll(result, shift * p, axis=-1))(
        jnp.arange(n_steps)
    )[..., p:-p]

    # Move the rolled dimension to the right position and reshape to get a batch of
    # square matrices.
    result = jnp.moveaxis(result, 0, -3).reshape((-1, n_steps * p, n_steps * p))

    # Set the last element to the innovation precision because there are no subsequent
    # samples. Set the first element to account for the different initial precision.
    result = ifnt.index_guard(result.at)[..., -p:, -p:].set(innovation_precision)
    result = ifnt.index_guard(result.at)[..., :p, :p].set(
        (negative_offdiag_block @ transition_matrix) + init_precision
    )
    # Restore the old batch shape.
    return result.reshape((*batch_shape, n_steps, p, n_steps, p))


class LinearDynamicalSystem(distributions.TransformedDistribution):
    """
    Linear dynamical system.

    .. math::

        y_{t + 1} = A y_t + x_t
    """

    pytree_aux_fields = ("n_steps",)
    arg_constraints = {
        "transition_matrix": distributions.constraints.real_matrix,
        "innovation_precision": distributions.constraints.positive_definite,
        "init_precision": distributions.constraints.positive_definite,
        "n_steps": distributions.constraints.positive_integer,
    }

    def __init__(
        self,
        transition_matrix: jnp.ndarray,
        innovation_precision: jnp.ndarray,
        init_precision: Optional[jnp.ndarray] = None,
        n_steps: int = 1,
        *,
        validate_args=None,
    ) -> None:
        self.transition_matrix = transition_matrix
        self.innovation_precision = innovation_precision
        if init_precision is None:
            init_precision = self.innovation_precision
        self.init_precision = init_precision
        self.n_steps = n_steps

        _, p = self.transition_matrix.shape
        innovation_distribution = distributions.MultivariateNormal(
            jnp.zeros(p),
            precision_matrix=jnp.concatenate(
                [
                    self.init_precision[..., None, :, :],
                    jnp.repeat(
                        self.innovation_precision[..., None, :, :],
                        self.n_steps - 1,
                        axis=-3,
                    ),
                ],
                axis=-3,
            ),
        )
        transform = distributions.transforms.RecursiveLinearTransform(
            self.transition_matrix
        )
        super().__init__(
            innovation_distribution, transform, validate_args=validate_args
        )

    @property
    def mean(self):
        return jnp.zeros(self.batch_shape + self.event_shape)

    @lazy_property
    def variance(self):
        t = jnp.arange(self.n_steps)
        i = jnp.arange(self.transition_matrix.shape[-1])
        return self.covariance_tensor[..., t, :, t, :][..., i, i]

    @lazy_property
    def covariance_tensor(self):
        precision_matrix = self.precision_tensor
        shape = precision_matrix.shape
        size = self.n_steps * self.transition_matrix.shape[-1]
        return jnp.linalg.inv(
            precision_matrix.reshape(shape[:-4] + (size, size))
        ).reshape(shape)

    @lazy_property
    def precision_tensor(self):
        return _evaluate_markov_precision(
            transition_matrix=self.transition_matrix,
            innovation_precision=self.innovation_precision,
            init_precision=self.init_precision,
            n_steps=self.n_steps,
        )


class Reshaped(distributions.Distribution):
    """
    Reshape the batch and event shapes of a distribution.

    Args:
        base_distribution: Distribution to reshape.
        shape: Target shape.
    """

    pytree_data_fields = ("base_dist",)

    def __init__(
        self,
        base_dist: distributions.Distribution,
        batch_shape: Optional[tuple] = None,
        event_shape: Optional[tuple] = None,
        *,
        validate_args=None,
    ) -> None:
        assert (
            batch_shape is not None or event_shape is not None
        ), "Need at least one of `batch_shape` and `event_shape`."
        if batch_shape is None:
            batch_shape = base_dist.batch_shape
        if event_shape is None:
            event_shape = base_dist.event_shape

        size = math.prod(batch_shape)
        base_size = math.prod(base_dist.batch_shape)
        if size != base_size:
            raise ValueError(
                f"Target batch shape {batch_shape} ({size} elements) and base "
                f"distribution batch shape {base_dist.batch_shape} ({base_size} "
                "elements) have inconsistent sizes."
            )

        size = math.prod(event_shape)
        base_size = math.prod(base_dist.event_shape)
        if size != base_size:
            raise ValueError(
                f"Target event shape {batch_shape} with {size} elements and base "
                f"distribution event shape {base_dist.batch_shape} with {base_size} "
                "elements have inconsistent sizes."
            )
        self.base_dist = base_dist

        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @lazy_property
    def mean(self):
        return self.base_dist.mean.reshape(self.shape())

    @lazy_property
    def variance(self):
        return self.base_dist.variance.reshape(self.shape())

    def sample(self, key, sample_shape=()):
        return self.base_dist.sample(key, sample_shape).reshape(
            sample_shape + self.shape()
        )

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        sample_shape = value.shape[
            : value.ndim - len(self.batch_shape) - self.event_dim
        ]
        base_shape = (
            sample_shape + self.base_dist.batch_shape + self.base_dist.event_shape
        )
        log_prob = self.base_dist.log_prob(value.reshape(base_shape))
        return log_prob.reshape(sample_shape + self.batch_shape)

    def entropy(self):
        return self.base_dist.entropy().reshape(self.batch_shape)
