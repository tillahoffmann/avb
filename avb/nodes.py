import contextlib
import functools
from jax import numpy as jnp
from numpyro import distributions
import operator
from typing import Any, Callable, Generic, Type, TypeVar
from typing_extensions import Self
from . import dispatch
from ._expect import expect, apply_substitutions


def _monkeypatched_instancecheck(cls, instance):
    # For LazyDistribution(cls, ...), this behaves exactly as if we'd passed in an
    # instance of cls.
    if isinstance(instance, LazyDistribution):
        return issubclass(instance.cls, cls)
    return type.__instancecheck__(cls, instance)


# Sanity check there is no custom implementation on the metaclass, patch it, and delete
# it.
DistributionMeta = distributions.distribution.DistributionMeta
assert DistributionMeta.__instancecheck__ is type.__instancecheck__
DistributionMeta.__instancecheck__ = _monkeypatched_instancecheck
del DistributionMeta


D = TypeVar("D", bound=distributions.Distribution)


class LazyDistribution(Generic[D]):

    LAZY = False

    def __init__(self, cls: Type[D], *args, **kwargs) -> None:
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        if self.LAZY:
            sample = LazySample(self, *args, **kwargs)
            if kwargs.get("sample_intermediates"):
                return sample, []
            return sample
        else:
            return self.instance(*args, **kwargs)

    @property
    def instance(self) -> D:
        args = [
            arg.instance if isinstance(arg, LazyDistribution) else arg
            for arg in self.args
        ]
        kwargs = {
            key: arg.instance if isinstance(arg, LazyDistribution) else arg
            for key, arg in self.kwargs.items()
        }
        return self.cls(*args, **kwargs)

    @classmethod
    @contextlib.contextmanager
    def lazy_trace(cls, lazy=True):
        """
        Execute a trace lazily to construct a graphical model.
        """
        previous = cls.LAZY
        cls.LAZY = lazy
        yield
        cls.LAZY = previous


class Node:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def __matmul__(self, other) -> Self:
        return Operator(operator.matmul, self, other)


class LazySample(Node):
    def __init__(self, distribution: LazyDistribution, *args, **kwargs) -> None:
        self.distribution = distribution
        super().__init__(*args, **kwargs)


class Operator(Node):
    def __init__(self, operation: Callable, *args, **kwargs) -> None:
        self.operation = operation
        super().__init__(*args, **kwargs)


@expect.register
@apply_substitutions
def _expect_operator(self: Operator, expr: Any = 1, substitutions=None) -> jnp.ndarray:
    # Dispatch from an operator instance to the specific operation we're evaluating.
    return _expect_operation(self.operation, self, expr, substitutions=substitutions)


@dispatch.valuedispatch
@dispatch.reraise_not_implemented_with_args
def _expect_operation(
    _, operator: Operator, expr: Any = 1, substitutions=None
) -> jnp.ndarray:
    raise NotImplementedError


@_expect_operation.register(operator.matmul)
@dispatch.reraise_not_implemented_with_args
@apply_substitutions
def _expect_operation_matmul(
    matmul, operator: Operator, expr: Any = 1, substitutions=None
) -> jnp.ndarray:
    sexpect = functools.partial(expect, substitutions=substitutions)
    args = [substitutions[arg] for arg in operator.args]
    if expr == 1:
        args = [sexpect(arg) for arg in args]
        return functools.reduce(matmul, args[1:], args[0])
    elif expr == 2:
        # TODO: implement this more generally. We currently only support fixed design
        # matrix on the left and stochastic vector on the right. Batching isn't
        # supported.
        assert len(args) == 2
        design, coef = args
        assert isinstance(design, jnp.ndarray)
        assert isinstance(coef, jnp.ndarray) or (
            isinstance(coef, distributions.Distribution) and coef.event_dim == 0
        )
        return jnp.square(design) @ sexpect(coef, 2)
    else:
        raise NotImplementedError
