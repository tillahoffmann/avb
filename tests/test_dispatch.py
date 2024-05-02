from avb import dispatch
import functools


def test_classdispatch() -> None:
    class A:
        pass

    class B(A):
        pass

    class C(B):
        pass

    @dispatch.classdispatch
    def do_class(x):
        return "base"

    do_class.register(A, lambda x: "a")
    do_class.register(B, lambda x: "b")

    @functools.singledispatch
    def do_instance(x):
        return "base"

    do_instance.register(A, lambda x: "a")
    do_instance.register(B, lambda x: "b")

    assert (do_class(C), do_instance(C())) == ("b", "b")
    assert (do_class(B), do_instance(B())) == ("b", "b")
    assert (do_class(A), do_instance(A())) == ("a", "a")
    assert (do_class(int), do_instance(3)) == ("base", "base")


def test_valuedispatch() -> None:
    @dispatch.valuedispatch
    def do(x):
        return "base"

    do.register("a", lambda x: "hello")
    do.register(1, lambda x: "world")

    assert do(None) == "base"
    assert do("a") == "hello"
    assert do(1) == "world"


def test_valuedispatch_with_key() -> None:
    @dispatch.valuedispatch(key=lambda x: 2 * x)
    def do(x):
        return "base"

    do.register("aa", lambda x: "hello")
    do.register(4, lambda x: "world")

    assert do("a") == "hello"
    assert do("aa") == "base"
    assert do(2) == "world"
