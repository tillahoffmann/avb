import functools


def classdispatch(func):
    """
    Dispatch on an explicit type.
    """
    instancedispatch = functools.singledispatch(func)

    @functools.wraps(func)
    def _classdispatch_wrapper(cls, *args, **kwargs):
        return instancedispatch.dispatch(cls)(cls, *args, **kwargs)

    _classdispatch_wrapper.register = instancedispatch.register
    return _classdispatch_wrapper


def valuedispatch(func):
    """
    Dispatch on a hashable value.
    """

    registry = {}

    @functools.wraps(func)
    def _valuedispatch_wrapper(value, *args, **kwargs):
        try:
            impl = registry[value]
        except KeyError:
            impl = func
        return impl(value, *args, **kwargs)

    def _valuedispatch_register(value, func=None):
        if func is None:
            return functools.partial(_valuedispatch_register, value)
        registry[value] = func
        return func

    _valuedispatch_wrapper.register = _valuedispatch_register
    _valuedispatch_wrapper.registry = registry
    return _valuedispatch_wrapper


def reraise_not_implemented_with_args(func):
    @functools.wraps(func)
    def _reraise_with_args_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NotImplementedError as ex:
            raise NotImplementedError(
                f"`{func.__name__}` is not implemented for args={args} and "
                f"kwargs={kwargs}."
            ) from ex

    return _reraise_with_args_wrapper
