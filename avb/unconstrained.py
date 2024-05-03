import functools
from numpyro import distributions
from typing import Type, TypeVar
from .dispatch import classdispatch
from .distributions import PrecisionNormal, Reshaped


@functools.singledispatch
def to_params(self: distributions.Distribution):
    raise NotImplementedError(self)


@to_params.register
def _to_params_precision_normal(self: PrecisionNormal):
    return {
        "loc": self.loc,
        "precision": self.precision,
    }, {}


@to_params.register
def _to_params_gamma(self: distributions.Gamma):
    return {
        "rate": self.rate,
        "concentration": self.concentration,
    }, {}


@to_params.register
def _to_params_lowrank_multivariate_normal(
    self: distributions.LowRankMultivariateNormal,
):
    return {
        "loc": self.loc,
        "cov_factor": self.cov_factor,
        "cov_diag": self.cov_diag,
    }, {}


@to_params.register
def _to_params_wishart(self: distributions.Wishart):
    return {
        "concentration": self.concentration,
        "rate_matrix": self.rate_matrix,
    }, {}


def _to_unconstrained(self: distributions.Distribution, arg_constraints=None) -> tuple:
    arg_constraints = arg_constraints or {}
    params, aux = to_params(self)
    unconstrained = {}
    for key, value in params.items():
        constraint = arg_constraints.get(key, self.arg_constraints[key])
        transform = distributions.transforms.biject_to(constraint).inv
        unconstrained[key] = transform(value)
    return unconstrained, aux


D = TypeVar("D", bound=distributions.Distribution)


def _from_unconstrained(
    cls: Type[D],
    unconstrained: dict,
    aux: dict,
    arg_constraints=None,
    validate_args=None,
) -> D:
    arg_constraints = arg_constraints or {}
    params = {}
    for key, value in unconstrained.items():
        constraint = arg_constraints.get(key, cls.arg_constraints[key])
        transform = distributions.transforms.biject_to(constraint)
        params[key] = transform(value)
    return cls(**params, **aux, validate_args=validate_args)


to_unconstrained = functools.singledispatch(_to_unconstrained)
from_unconstrained = classdispatch(_from_unconstrained)


@to_unconstrained.register
def _to_unconstrained_wishart(self: distributions.Wishart, arg_constraints=None):
    _, p = self.event_shape
    arg_constraints.setdefault(
        "concentration", distributions.constraints.greater_than(p - 1)
    )
    return _to_unconstrained(self, arg_constraints)


@from_unconstrained.register(distributions.Wishart)
def _from_unconstrained_wishart(
    cls, unconstrained, aux, arg_constraints=None, validate_args=None
):
    p = unconstrained["rate_matrix"].shape[-1]
    arg_constraints.setdefault(
        "concentration", distributions.constraints.greater_than(p - 1)
    )
    return _from_unconstrained(
        cls, unconstrained, aux, arg_constraints, validate_args=validate_args
    )


def _to_unconstrained_base_dist(base_dist, arg_constraints):
    """
    Convert a distribution to a "base" representation that can be used by transformed
    distributions.
    """
    unconstrained, aux = to_unconstrained(base_dist)
    return {"base": unconstrained}, {"base": aux, "base_cls": base_dist.__class__}


def _from_unconstrained_base_dist(
    unconstrained, aux, arg_constraints=None, validate_args=None
):
    return from_unconstrained(
        aux.pop("base_cls"),
        unconstrained.pop("base"),
        aux.pop("base"),
        arg_constraints=arg_constraints,
        validate_args=validate_args,
    )


@from_unconstrained.register(Reshaped)
def _from_unconstrained_reshaped(
    cls, unconstrained, aux, arg_constraints=None, validate_args=None
):
    base_dist = _from_unconstrained_base_dist(
        unconstrained, aux, arg_constraints, validate_args=validate_args
    )
    return cls(base_dist, **aux)


@to_unconstrained.register
def _to_unconstrained_reshaped(self: Reshaped, arg_constraints=None):
    unconstrained, aux = _to_unconstrained_base_dist(self.base_dist, arg_constraints)
    aux.update(
        {
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
            "reinterpreted_batch_ndims": self.reinterpreted_batch_ndims,
        }
    )
    return unconstrained, aux


@from_unconstrained.register(distributions.Independent)
def _from_unconstrained_independent(
    cls, unconstrained, aux, arg_constraints=None, validate_args=None
):
    base_dist = _from_unconstrained_base_dist(
        unconstrained, aux, arg_constraints, validate_args=validate_args
    )
    return cls(base_dist, **aux)
