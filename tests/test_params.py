import numpy as np

from whitenn.runtime.params import ParamError, ParamStore, init_normal, init_zeros


def test_paramstore_add_param_zeros():
    store = ParamStore(seed=123)
    param = store.add_param("W", [2, 3], init=init_zeros)

    assert param.name == "W"
    assert param.shape == (2, 3)
    assert param.value.shape == (2, 3)
    assert np.all(param.value.as_array() == 0.0)


def test_paramstore_add_param_normal():
    store = ParamStore(seed=0)
    param = store.add_param("b", [4], init=init_normal)

    assert param.shape == (4,)
    assert param.value.shape == (4,)
    assert np.any(param.value.as_array() != 0.0)


def test_paramstore_duplicate_name():
    store = ParamStore()
    store.add_param("W", [1], init=init_zeros)
    try:
        store.add_param("W", [1], init=init_zeros)
    except ParamError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("Expected ParamError for duplicate param")
