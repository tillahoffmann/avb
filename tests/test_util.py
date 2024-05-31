from avb import util


def test_apply_scale():
    assert util.apply_scale(
        {"x": [3, 4]},
        {"x": [5]},
    ) == {"x": [15, 4]}


def test_precondition_diagonal():
    assert util.precondition_diagonal(lambda x: x**2, 2)(3) == 36
