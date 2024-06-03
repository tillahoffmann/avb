from avb import util
import numpy as np


def test_apply_scale():
    assert util.apply_scale(
        {"x": [3, 4]},
        {"x": [5]},
    ) == {"x": [15, 4]}


def test_precondition_diagonal():
    assert util.precondition_diagonal(lambda x: x**2, 2)(3) == 36


def test_hessian_diagonal_finite_diff():
    np.testing.assert_allclose(
        util.hessian_diagonal_finite_diff(lambda x, a: a * x**2, 1, 3, eps=1e-8), 6
    )
