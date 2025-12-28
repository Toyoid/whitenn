import numpy as np

from whitenn.runtime.autodiff import Grad
from whitenn.runtime.graph import Graph
from whitenn.runtime.optim import OptimizerError, SGD
from whitenn.runtime.params import ParamStore, init_zeros
from whitenn.runtime.values import Value


def test_sgd_updates_param():
    params = ParamStore(seed=0)
    params.add_param("W", [2], init=init_zeros)
    params.get("W").value = Value(np.array([1.0, -2.0]))

    grad = Grad(
        grads={"W": Value(np.array([0.5, -1.0]))},
        loss="y",
        graph=Graph([], {}),
        node_grads={},
        trace=[],
    )
    SGD(lr=0.1).apply(grad, params)

    assert np.allclose(params.get("W").value.as_array(), np.array([0.95, -1.9]))


def test_sgd_skips_nontrainable():
    params = ParamStore(seed=0)
    params.add_param("b", None, init=init_zeros, trainable=False)
    params.get("b").value = Value(2.0)

    grad = Grad(grads={"b": Value(1.0)}, loss="y", graph=Graph([], {}), node_grads={}, trace=[])
    SGD(lr=0.5).apply(grad, params)

    assert params.get("b").value.as_array() == 2.0


def test_sgd_shape_mismatch():
    params = ParamStore(seed=0)
    params.add_param("W", [2], init=init_zeros)
    params.get("W").value = Value(np.array([1.0, 2.0]))

    grad = Grad(
        grads={"W": Value(np.array([1.0, 2.0, 3.0]))},
        loss="y",
        graph=Graph([], {}),
        node_grads={},
        trace=[],
    )
    try:
        SGD(lr=0.1).apply(grad, params)
    except OptimizerError as exc:
        assert "shape mismatch" in str(exc)
    else:
        raise AssertionError("Expected OptimizerError for shape mismatch")
