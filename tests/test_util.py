from avb import util
import ifnt
import jax
from jax import numpy as jnp
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


def test_hessdiag():
    rng = ifnt.random.JaxRandomState(17)
    arg = {
        "params": {
            "loc": rng.normal((30,)),
            "scale": rng.gamma(3, (30,)),
        },
        "x": rng.normal((30,)),
    }

    def func(arg):
        z = (arg["x"] - arg["params"]["loc"]) / arg["params"]["scale"]
        return jnp.square(z).sum()

    hessdiag = jax.jit(util.hessdiag(func))(arg)
    hess = jax.jacfwd(jax.jacrev(func))(arg)

    # Compare the results.
    for path, value in jax.tree_util.tree_flatten_with_path(hessdiag)[0]:
        # Get the corresponding full Hessian and extract diagonal elements.
        reference = util.tree_get(hess, path * 2)
        reference = jnp.diagonal(reference.reshape((value.size, value.size))).reshape(
            value.shape
        )
        np.testing.assert_allclose(value, reference, rtol=1e-6)

    # Sanity checks.
    np.testing.assert_allclose(hessdiag["x"], hessdiag["params"]["loc"], rtol=1e-6)
    np.testing.assert_allclose(
        hessdiag["x"], 2 / arg["params"]["scale"] ** 2, rtol=1e-6
    )
