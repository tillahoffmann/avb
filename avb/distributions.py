from jax import numpy as jnp
from numpyro import distributions


class PrecisionNormal(distributions.Normal):
    arg_constraints = {
        "loc": distributions.constraints.real,
        "precision": distributions.constraints.positive,
    }

    def __init__(self, loc=0.0, precision=1.0, *, validate_args=None) -> None:
        super().__init__(
            loc=loc, scale=1.0 / jnp.sqrt(precision), validate_args=validate_args
        )
