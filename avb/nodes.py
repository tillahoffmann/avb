import contextlib
from numpyro import distributions
import operator
from typing import Callable, Generic, Type, TypeVar
from typing_extensions import Self


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

    def __add__(self, other) -> Self:
        return Operator(operator.add, self, other)


class LazySample(Node):
    def __init__(self, distribution: LazyDistribution, *args, **kwargs) -> None:
        self.distribution = distribution
        super().__init__(*args, **kwargs)


class Operator(Node):
    def __init__(self, operation: Callable, *args, **kwargs) -> None:
        self.operation = operation
        super().__init__(*args, **kwargs)
