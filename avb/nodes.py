from jax import numpy as jnp
import numbers
from numpyro.handlers import Messenger
from numpyro import distributions
import operator
from typing import Generic, Type, TypeVar
from .dispatch import valuedispatch


def _monkeypatched_instancecheck(cls, instance):
    # For LazyDistribution(cls, ...), this behaves exactly as if we'd passed in an
    # instance of cls.
    if isinstance(instance, DelayedDistribution):
        return issubclass(instance.cls, cls)
    return type.__instancecheck__(cls, instance)


# Sanity check there is no custom implementation on the metaclass, patch it, and delete
# it.
DistributionMeta = distributions.distribution.DistributionMeta
assert DistributionMeta.__instancecheck__ is type.__instancecheck__
DistributionMeta.__instancecheck__ = _monkeypatched_instancecheck
del DistributionMeta


D = TypeVar("D", bound=distributions.Distribution)


class DelayedDistribution(Generic[D]):

    def __init__(self, cls: Type[D], *args, **kwargs) -> None:
        self.cls = cls
        self.args = args
        self.kwargs = kwargs
        self._instance = None

    def __call__(self, *args, **kwargs):
        return self.instance(*args, **kwargs)

    @property
    def instance(self) -> D:
        if self._instance is None:
            self._instance = self.cls(*self.args, **self.kwargs)
        return self._instance

    def __repr__(self):
        return f"{self.__class__.__name__}({self.cls.__name__}, ...)"

    @property
    def shape(self) -> tuple:
        batch_shape, event_shape = self.cls.infer_shapes(
            *(
                () if isinstance(arg, numbers.Number) else arg.shape
                for arg in self.args
            ),
            **{
                key: () if isinstance(arg, numbers.Number) else arg.shape
                for key, arg in self.kwargs.items()
            },
        )
        return batch_shape + event_shape


class Node:
    def __matmul__(self, other):
        return Operator(operator.matmul, self, other)

    def __add__(self, other):
        return Operator(operator.add, self, other)

    def __getitem__(self, other):
        return Operator(operator.getitem, self, other)


class Operator(Node):
    def __init__(self, operator, *args, **kwargs):
        self.operator = operator
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        return f"{self.__class__.__name__}({self.operator.__name__}, ...)"

    @property
    def shape(self) -> tuple:
        return infer_operator_shape(
            self.operator,
            *(
                () if isinstance(arg, numbers.Number) else arg.shape
                for arg in self.args
            ),
            **{
                key: () if isinstance(arg, numbers.Number) else arg.shape
                for key, arg in self.kwargs.items()
            },
        )


@valuedispatch
def infer_operator_shape(self, *args, **kwargs) -> tuple:
    raise NotImplementedError(self)


@infer_operator_shape.register(operator.add)
def _infer_operator_add_shape(self, *args, **kwargs) -> tuple:
    return jnp.broadcast_shapes(*args)


@infer_operator_shape.register(operator.matmul)
def _infer_operator_matmul_shape(self, *args, **kwargs) -> tuple:
    a, b = args
    assert len(a) == 2 and len(b) == 1 and a[-1] == b[-1]
    return a[:1]


class DelayedValue(Node):
    def __init__(self, value=None, name=None, shape=None):
        assert not isinstance(value, Node), "Delayed values cannot be nested."
        self._value = value
        self.name = name
        self.shape = shape
        if value is not None and shape is not None:
            if isinstance(value, distributions.Distribution):
                value_shape = value.shape()
            elif isinstance(value, numbers.Number):
                value_shape = ()
            else:
                value_shape = value.shape
            assert value_shape == shape

    @property
    def has_value(self):
        return self._value is not None

    @property
    def value(self):
        if not self.has_value:
            raise RuntimeError("Delayed value has no value.")
        return self._value

    def __repr__(self):
        value = self.value
        if isinstance(value, jnp.ndarray):
            value = f"Array(shape={value.shape}, dtype={value.dtype})"
        elif isinstance(value, distributions.Distribution):
            value = (
                f"{value.__class__.__name__}(batch_shape={value.batch_shape}, "
                f"event_shape={value.event_shape})"
            )
        cls = self.__class__.__name__
        if self.name:
            return f"{cls}('{self.name}', value={value})"
        return f"{cls}(value={value})"

    @classmethod
    def materialize(cls, arg, *args):
        arg = arg.value if isinstance(arg, cls) else arg
        if not args:
            return arg
        return (arg, *(x.value if isinstance(x, cls) else x for x in args))


class delay(Messenger):
    def process_message(self, msg):
        fn: DelayedDistribution = msg["fn"]
        shape = msg["kwargs"].get("sample_shape", ()) + fn.shape
        msg["value"] = DelayedValue(msg["value"], name=msg["name"], shape=shape)
