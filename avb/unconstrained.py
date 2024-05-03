import functools
from numpyro import distributions
from typing import Tuple
from .dispatch import valuedispatch
from .distributions import PrecisionNormal, Reshaped


@functools.singledispatch
def to_params(self: distributions.Distribution) -> Tuple[dict, dict]:
    """
    Convert a distribution to a tuple of differentiable and auxiliary parameters,
    including the class of the distribution for reconstruction.
    """
    raise NotImplementedError(self)


@to_params.register
def _to_params_precision_normal(self: PrecisionNormal):
    return {
        "loc": self.loc,
        "precision": self.precision,
    }, {"cls": self.__class__}


@to_params.register
def _to_params_gamma(self: distributions.Gamma):
    return {
        "rate": self.rate,
        "concentration": self.concentration,
    }, {"cls": self.__class__}


@to_params.register
def _to_params_lowrank_multivariate_normal(
    self: distributions.LowRankMultivariateNormal,
):
    return {
        "loc": self.loc,
        "cov_factor": self.cov_factor,
        "cov_diag": self.cov_diag,
    }, {"cls": self.__class__}


@to_params.register
def _to_params_wishart(self: distributions.Wishart):
    return {
        "concentration": self.concentration,
        "rate_matrix": self.rate_matrix,
    }, {"cls": self.__class__}


@functools.singledispatch
def to_unconstrained(
    self: distributions.Distribution, arg_constraints=None
) -> Tuple[dict, dict]:
    """
    Transform a distribution to a tuple of optimizable, unconstrained parameters and
    static auxiliary information.

    Args:
        self: Distribution to transform.
        arg_constraints: Argument constraints overriding the default constraints of the
            distribution, e.g., for dependent constraints.
    """
    arg_constraints = arg_constraints or {}
    params, aux = to_params(self)
    unconstrained = {}
    for key, value in params.items():
        constraint = arg_constraints.get(key, self.arg_constraints[key])
        transform = distributions.transforms.biject_to(constraint).inv
        unconstrained[key] = transform(value)
    return unconstrained, aux


@valuedispatch(key=lambda x: x["cls"], argnum=1)
def from_unconstrained(
    unconstrained: dict,
    aux: dict,
    arg_constraints=None,
    *,
    validate_args=None,
) -> distributions.Distribution:
    """
    Transform a tuple of unconstrained parameters and static auxiliary information to a
    distribution instance.

    Args:
        unconstrained: Unconstrained parameters.
        aux: Static auxiliary information, including the distribution type a the
            :code:`cls` key.
        arg_constraints: Argument constraints overriding the default constraints of the
            distribution, e.g., for dependent constraints.
        validate_args: Validate the distribution arguments.

    Returns:
        Distribution instance.
    """
    cls = aux.pop("cls")
    arg_constraints = arg_constraints or {}
    params = {}
    for key, value in unconstrained.items():
        constraint = arg_constraints.get(key, cls.arg_constraints[key])
        transform = distributions.transforms.biject_to(constraint)
        params[key] = transform(value)
    return cls(**params, **aux, validate_args=validate_args)


@to_unconstrained.register
def _to_unconstrained_wishart(self: distributions.Wishart, arg_constraints=None):
    _, p = self.event_shape
    arg_constraints.setdefault(
        "concentration", distributions.constraints.greater_than(p - 1)
    )
    return to_unconstrained.__wrapped__(self, arg_constraints)


@from_unconstrained.register(distributions.Wishart)
def _from_unconstrained_wishart(
    unconstrained, aux, arg_constraints=None, *, validate_args=None
):
    p = unconstrained["rate_matrix"].shape[-1]
    arg_constraints.setdefault(
        "concentration", distributions.constraints.greater_than(p - 1)
    )
    return from_unconstrained.__wrapped__(
        unconstrained, aux, arg_constraints, validate_args=validate_args
    )


def _to_unconstrained_base_dist(base_dist, arg_constraints=None):
    """
    Convert a distribution to a "base" representation that can be used by transformed
    distributions, e.g., :code:`Reshaped` or `numpyro.distributions.Independent`.
    """
    unconstrained, aux = to_unconstrained(base_dist, arg_constraints=arg_constraints)
    return {"base": unconstrained}, {"base": aux}


def _from_unconstrained_base_dist(
    unconstrained, aux, arg_constraints=None, *, validate_args=None
) -> distributions.Distribution:
    return from_unconstrained(
        unconstrained.pop("base"),
        aux.pop("base"),
        arg_constraints=arg_constraints,
        validate_args=validate_args,
    )


@to_unconstrained.register
def _to_unconstrained_reshaped(self: Reshaped, arg_constraints=None):
    unconstrained, aux = _to_unconstrained_base_dist(self.base_dist, arg_constraints)
    aux.update(
        {
            "cls": self.__class__,
            "batch_shape": self.batch_shape,
            "event_shape": self.event_shape,
        }
    )
    return unconstrained, aux


@to_unconstrained.register
def _to_unconstrained_independent(
    self: distributions.Independent, arg_constraints=None
):
    unconstrained, aux = _to_unconstrained_base_dist(self.base_dist, arg_constraints)
    aux.update(
        {
            "cls": self.__class__,
            "reinterpreted_batch_ndims": self.reinterpreted_batch_ndims,
        }
    )
    return unconstrained, aux


@from_unconstrained.register(distributions.Independent)
@from_unconstrained.register(Reshaped)
def _from_unconstrained_with_base_dist(
    unconstrained, aux, arg_constraints=None, *, validate_args=None
):
    base_dist = _from_unconstrained_base_dist(
        unconstrained, aux, arg_constraints, validate_args=validate_args
    )
    return aux.pop("cls")(base_dist, **aux)
