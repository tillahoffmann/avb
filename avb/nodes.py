from jax import numpy as jnp
import numbers
from numpyro.handlers import Messenger
from numpyro import distributions
import operator
from typing import Generic, Type, TypeVar
from .dispatch import classdispatch
from .util import get_shape


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
        self._batch_shape = None
        self._event_shape = None

    def __call__(self, *args, **kwargs):
        return self.instance(*args, **kwargs)

    @property
    def instance(self) -> D:
        if self._instance is None:
            self._instance = self.cls(*self.args, **self.kwargs)
        return self._instance

    def __repr__(self):
        return f"{self.__class__.__name__}({self.cls.__name__}, ...)"

    def shape(self, sample_shape=()) -> tuple:
        return sample_shape + self.batch_shape + self.event_shape

    @property
    def batch_shape(self):
        if self._batch_shape is None:
            self._infer_shapes()
        return self._batch_shape

    @property
    def event_shape(self):
        if self._event_shape is None:
            self._infer_shapes()
        return self._event_shape

    def _infer_shapes(self) -> tuple:
        self._batch_shape, self._event_shape = delay_infer_shapes(
            self.cls, *self.args, **self.kwargs
        )
        return self._batch_shape, self._event_shape


@classdispatch
def delay_infer_shapes(cls: distributions.Distribution, *args, **kwargs):
    return cls.infer_shapes(
        *(get_shape(arg) for arg in args),
        **{key: get_shape(arg) for key, arg in kwargs.items()},
    )


class Node:
    def __matmul__(self, other):
        return MatMulOperator(self, other)

    def __rmatmul__(self, other):
        return MatMulOperator(other, self)

    def __add__(self, other):
        return AddOperator(self, other)

    def __radd__(self, other):
        return AddOperator(other, self)

    def __getitem__(self, other):
        return GetItemOperator(self, other)

    def __mul__(self, other):
        return MulOperator(self, other)

    def __rmul__(self, other):
        return MulOperator(other, self)

    def sum(self, axis=None):
        return SumOperator(self, axis=axis)


class Operator(Node):
    operator = None

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @property
    def shape(self) -> tuple:
        raise NotImplementedError


class MatMulOperator(Operator):
    operator = operator.matmul

    @property
    def shape(self) -> tuple:
        a, b = map(get_shape, self.args)
        assert len(a) == 2 and len(b) == 1 and a[-1] == b[-1]
        return a[:1]


class AddOperator(Operator):
    operator = operator.add

    @property
    def shape(self) -> tuple:
        return jnp.broadcast_shapes(*map(get_shape, self.args))


class GetItemOperator(Operator):
    operator = operator.getitem

    @property
    def shape(self) -> tuple:
        a, b = self.args
        return jnp.empty(a.shape)[b].shape


class MulOperator(Operator):
    operator = operator.mul

    @property
    def shape(self) -> tuple:
        return jnp.broadcast_shapes(*map(get_shape, self.args))


class SumOperator(Operator):
    operator = jnp.sum

    @property
    def shape(self) -> tuple:
        return jnp.empty(self.args[0].shape).sum(**self.kwargs).shape


class DelayedValue(Node):
    def __init__(self, value=None, name=None, shape=None):
        assert not isinstance(value, Node), "Delayed values cannot be nested."
        self._value = value
        self.name = name
        if value is not None:
            # Get the shape of the value.
            if isinstance(value, distributions.Distribution):
                value_shape = value.shape()
            elif isinstance(value, numbers.Number):
                value_shape = ()
            else:
                value_shape = value.shape

            # Infer the shape if not given.
            if shape is None:
                shape = value_shape
            # Compare with the expected shape if given.
            elif value_shape != shape:
                raise ValueError(
                    f"Expected shape `{shape}` but got `{value_shape}` for parameter "
                    f"named `{self.name}`."
                )

        self.shape = shape

    @property
    def has_value(self):
        return self._value is not None

    @property
    def value(self):
        if not self.has_value:
            raise RuntimeError("Delayed value has no value.")
        return self._value

    def __repr__(self):
        if self.has_value:
            value = self.value
        else:
            value = None
        if isinstance(value, jnp.ndarray):
            value = f"Array(shape={value.shape}, dtype={value.dtype})"
        elif isinstance(value, distributions.Distribution):
            value = (
                f"{value.__class__.__name__}(batch_shape={value.batch_shape}, "
                f"event_shape={value.event_shape})"
            )
        else:
            value = self.value
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
    """
    Delay sampling to trace parameters through the model.
    """

    def process_message(self, msg):
        fn = msg["fn"]
        assert isinstance(
            fn, DelayedDistribution
        ), f"Cannot delay sampling from {fn} because it is not a `DelayedDistribution`."
        shape = msg["kwargs"].get("sample_shape", ()) + fn.shape()
        msg["value"] = DelayedValue(msg["value"], name=msg["name"], shape=shape)


class materialize(Messenger):
    """
    Materialize `DelayedDistribution`s to recover an equivalent numpyro model.
    """

    def process_message(self, msg):
        fn = msg["fn"]
        assert isinstance(
            fn, DelayedDistribution
        ), f"Cannot materialize {fn} because it is not a `DelayedDistribution`."
        msg["fn"] = fn.instance
