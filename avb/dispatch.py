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


def valuedispatch(func=None, *, key=None):
    """
    Dispatch on a hashable value.
    """

    if func is None:
        return functools.partial(valuedispatch, key=key)

    registry = {}

    @functools.wraps(func)
    def _valuedispatch_wrapper(value, *args, **kwargs):
        try:
            impl = registry[value if key is None else key(value)]
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
