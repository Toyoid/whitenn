import numpy as np
import pytest

from whitenn.parser import parse_program
from whitenn.runtime.autodiff import derive
from whitenn.runtime.graph import GraphExecutor
from whitenn.runtime.params import ParamStore, init_zeros
from whitenn.runtime.rules import RuleTable
from whitenn.runtime.values import Value


def test_autodiff_binop_mul():
    source = """
    model M {
      param W init zeros;
    }

    fn f() {
      graph {
        y = x * W;
      }
    }
    """
    program = parse_program(source)
    rules = RuleTable.from_program(program)
    params = ParamStore(seed=0)
    params.add_param("W", None, init=init_zeros)
    params.get("W").value = Value(3.0)

    graph_stmt = program.items[1].body.stmts[0]
    graph = GraphExecutor(rules, params).execute(graph_stmt, env={"x": 2.0})
    grad = derive(graph, "y", ["W"], rules)

    assert grad.grads["W"].as_array() == 2.0


def test_autodiff_rule_call():
    source = """
    model M {
      param W init zeros;
    }

    rule add(x: Real, y: Real) : Real {
      forward = x + y;
      d/dx = 1;
      d/dy = 1;
    }

    fn f() {
      graph {
        y = add(x, W);
      }
    }
    """
    program = parse_program(source)
    rules = RuleTable.from_program(program)
    params = ParamStore(seed=0)
    params.add_param("W", None, init=init_zeros)
    params.get("W").value = Value(0.5)

    graph_stmt = program.items[2].body.stmts[0]
    graph = GraphExecutor(rules, params).execute(graph_stmt, env={"x": 1.0})
    grad = derive(graph, "y", ["W"], rules)

    assert grad.grads["W"].as_array() == 1.0


def test_autodiff_runtime_error_has_location():
    source = """
    model M {
      param W[2,2] init zeros;
    }

    fn f(x[2]) {
      graph {
        y = x @ W;
      }
    }
    """
    program = parse_program(source, filename="bad.wnn")
    rules = RuleTable.from_program(program)
    params = ParamStore(seed=0)
    params.add_param("W", [2, 2], init=init_zeros)

    graph_stmt = program.items[1].body.stmts[0]
    with pytest.raises(Exception) as excinfo:
        GraphExecutor(rules, params, source=program.source, filename=program.filename).execute(
            graph_stmt, env={"x": np.array([1.0, 2.0, 3.0])}
        )
    assert "bad.wnn" in str(excinfo.value)


def test_autodiff_exp_builtin():
    source = """
    model M {
      param W init zeros;
    }

    fn f() {
      graph {
        y = exp(W);
      }
    }
    """
    program = parse_program(source)
    rules = RuleTable.from_program(program)
    params = ParamStore(seed=0)
    params.add_param("W", None, init=init_zeros)
    params.get("W").value = Value(1.5)

    graph_stmt = program.items[1].body.stmts[0]
    graph = GraphExecutor(rules, params).execute(graph_stmt)
    grad = derive(graph, "y", ["W"], rules)

    expected = np.exp(1.5)
    assert np.allclose(grad.grads["W"].as_array(), expected)


def test_autodiff_chain_relu():
    source = """
    model M {
      param W init zeros;
      param b init zeros;
    }

    rule relu(x: Real) : Real {
      forward = max(0, x);
      d/dx = (x > 0) ? 1 : 0;
    }

    fn f() {
      graph {
        y = relu(x * W + b);
      }
    }
    """
    program = parse_program(source)
    rules = RuleTable.from_program(program)
    params = ParamStore(seed=0)
    params.add_param("W", None, init=init_zeros)
    params.add_param("b", None, init=init_zeros)
    params.get("W").value = Value(3.0)
    params.get("b").value = Value(-1.0)

    graph_stmt = program.items[2].body.stmts[0]
    graph = GraphExecutor(rules, params).execute(graph_stmt, env={"x": 2.0})
    grad = derive(graph, "y", ["W", "b"], rules)

    assert grad.grads["W"].as_array() == 2.0
    assert grad.grads["b"].as_array() == 1.0


def test_autodiff_matmul_matrix():
    source = """
    model M {
      param W[2,2] init zeros;
    }

    fn f(x[2,2]) {
      graph {
        y = x @ W;
      }
    }
    """
    program = parse_program(source)
    rules = RuleTable.from_program(program)
    params = ParamStore(seed=0)
    params.add_param("W", [2, 2], init=init_zeros)
    params.get("W").value = Value(np.array([[1.0, 2.0], [3.0, 4.0]]))

    graph_stmt = program.items[1].body.stmts[0]
    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    graph = GraphExecutor(rules, params).execute(graph_stmt, env={"x": x})
    grad = derive(graph, "y", ["W"], rules)

    expected = x.T @ np.ones_like(graph.get("y").value.as_array())
    assert np.allclose(grad.grads["W"].as_array(), expected)


def test_autodiff_matmul_vector_matrix():
    source = """
    model M {
      param W[2,3] init zeros;
    }

    fn f(x[2]) {
      graph {
        y = x @ W;
      }
    }
    """
    program = parse_program(source)
    rules = RuleTable.from_program(program)
    params = ParamStore(seed=0)
    params.add_param("W", [2, 3], init=init_zeros)
    params.get("W").value = Value(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    graph_stmt = program.items[1].body.stmts[0]
    x = np.array([1.0, -2.0])
    graph = GraphExecutor(rules, params).execute(graph_stmt, env={"x": x})
    grad = derive(graph, "y", ["W"], rules)

    expected = np.outer(x, np.ones_like(graph.get("y").value.as_array()))
    assert np.allclose(grad.grads["W"].as_array(), expected)


def test_autodiff_broadcast_add_reduces_grad():
    source = """
    model M {
      param b[3] init zeros;
    }

    fn f(x[1,3]) {
      graph {
        y = x + b;
      }
    }
    """
    program = parse_program(source)
    rules = RuleTable.from_program(program)
    params = ParamStore(seed=0)
    params.add_param("b", [3], init=init_zeros)
    params.get("b").value = Value(np.array([0.0, 0.0, 0.0]))

    graph_stmt = program.items[1].body.stmts[0]
    x = np.array([[1.0, 2.0, 3.0]])
    graph = GraphExecutor(rules, params).execute(graph_stmt, env={"x": x})
    grad = derive(graph, "y", ["b"], rules)

    assert np.allclose(grad.grads["b"].as_array(), np.ones(3))


def test_autodiff_softmax_finite_difference():
    source = """
    model M {
      param x[3] init zeros;
    }

    fn f() {
      graph {
        y = softmax(x);
        L = sum(y * [1.0, 2.0, 3.0]);
      }
    }
    """
    program = parse_program(source)
    rules = RuleTable.from_program(program)
    params = ParamStore(seed=0)
    params.add_param("x", [3], init=init_zeros)
    graph_stmt = program.items[1].body.stmts[0]
    executor = GraphExecutor(rules, params)

    def loss_for(vec):
        params.get("x").value = Value(np.array(vec))
        graph = executor.execute(graph_stmt)
        return float(graph.get("L").value.as_array())

    x0 = np.array([0.2, -0.1, 0.4])
    params.get("x").value = Value(x0)
    graph = executor.execute(graph_stmt)
    grad = derive(graph, "L", ["x"], rules)
    grad_x = grad.grads["x"].as_array()

    eps = 1e-5
    fd = np.zeros_like(x0)
    for i in range(len(x0)):
        xp = x0.copy()
        xm = x0.copy()
        xp[i] += eps
        xm[i] -= eps
        fd[i] = (loss_for(xp) - loss_for(xm)) / (2 * eps)

    assert np.allclose(grad_x, fd, atol=1e-4, rtol=1e-4)


def test_autodiff_transpose_sum_gradient():
    source = """
    model M {
      param X[2,3] init zeros;
    }

    fn f() {
      graph {
        L = sum(transpose(X) * [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
      }
    }
    """
    program = parse_program(source)
    rules = RuleTable.from_program(program)
    params = ParamStore(seed=0)
    params.add_param("X", [2, 3], init=init_zeros)
    params.get("X").value = Value(np.array([[0.1, -0.2, 0.3], [0.0, 0.4, -0.1]]))

    graph_stmt = program.items[1].body.stmts[0]
    graph = GraphExecutor(rules, params).execute(graph_stmt)
    grad = derive(graph, "L", ["X"], rules)

    expected = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
    assert np.allclose(grad.grads["X"].as_array(), expected)


def test_autodiff_log_finite_difference():
    source = """
    model M {
      param x[3] init zeros;
    }

    fn f() {
      graph {
        L = sum(log(x));
      }
    }
    """
    program = parse_program(source)
    rules = RuleTable.from_program(program)
    params = ParamStore(seed=0)
    params.add_param("x", [3], init=init_zeros)
    graph_stmt = program.items[1].body.stmts[0]
    executor = GraphExecutor(rules, params)

    def loss_for(vec):
        params.get("x").value = Value(np.array(vec))
        graph = executor.execute(graph_stmt)
        return float(graph.get("L").value.as_array())

    x0 = np.array([0.5, 1.5, 2.0])
    params.get("x").value = Value(x0)
    graph = executor.execute(graph_stmt)
    grad = derive(graph, "L", ["x"], rules)
    grad_x = grad.grads["x"].as_array()

    eps = 1e-5
    fd = np.zeros_like(x0)
    for i in range(len(x0)):
        xp = x0.copy()
        xm = x0.copy()
        xp[i] += eps
        xm[i] -= eps
        fd[i] = (loss_for(xp) - loss_for(xm)) / (2 * eps)

    assert np.allclose(grad_x, fd, atol=1e-4, rtol=1e-4)


def test_autodiff_cross_entropy_rule():
    source = """
    model M {
      param P[3] init zeros;
    }

    rule cross_entropy(p: Real, t: Real nondiff) : Real {
      forward = -t * log(p);
      d/dp = -t / p;
    }

    fn f(t[3]) {
      graph {
        L = sum(cross_entropy(P, t));
      }
    }
    """
    program = parse_program(source)
    rules = RuleTable.from_program(program)
    params = ParamStore(seed=0)
    params.add_param("P", [3], init=init_zeros)
    params.get("P").value = Value(np.array([0.2, 0.5, 0.3]))

    graph_stmt = program.items[2].body.stmts[0]
    graph = GraphExecutor(rules, params).execute(graph_stmt, env={"t": np.array([0.0, 1.0, 0.0])})
    grad = derive(graph, "L", ["P"], rules)

    assert np.allclose(grad.grads["P"].as_array(), np.array([0.0, -2.0, 0.0]))


def test_autodiff_finite_difference_scalar():
    source = """
    model M {
      param W init zeros;
      param b init zeros;
    }

    fn f() {
      graph {
        y = exp(x * W + b);
      }
    }
    """
    program = parse_program(source)
    rules = RuleTable.from_program(program)
    params = ParamStore(seed=0)
    params.add_param("W", None, init=init_zeros)
    params.add_param("b", None, init=init_zeros)

    graph_stmt = program.items[1].body.stmts[0]
    executor = GraphExecutor(rules, params)

    x_value = 1.3
    eps = 1e-5

    def loss_for(w, b):
        params.get("W").value = Value(w)
        params.get("b").value = Value(b)
        graph = executor.execute(graph_stmt, env={"x": x_value})
        return float(graph.get("y").value.as_array())

    w0 = 0.7
    b0 = -0.4
    base = loss_for(w0, b0)

    graph = executor.execute(graph_stmt, env={"x": x_value})
    grad = derive(graph, "y", ["W", "b"], rules)
    grad_w = float(grad.grads["W"].as_array())
    grad_b = float(grad.grads["b"].as_array())

    fd_w = (loss_for(w0 + eps, b0) - loss_for(w0 - eps, b0)) / (2 * eps)
    fd_b = (loss_for(w0, b0 + eps) - loss_for(w0, b0 - eps)) / (2 * eps)

    assert np.isfinite(base)
    assert np.allclose(grad_w, fd_w, rtol=1e-4, atol=1e-6)
    assert np.allclose(grad_b, fd_b, rtol=1e-4, atol=1e-6)


def test_autodiff_finite_difference_matrix():
    source = """
    model M {
      param W[2,2] init zeros;
    }

    fn f(x[2,2]) {
      graph {
        y = x @ W;
      }
    }
    """
    program = parse_program(source)
    rules = RuleTable.from_program(program)
    params = ParamStore(seed=0)
    params.add_param("W", [2, 2], init=init_zeros)

    graph_stmt = program.items[1].body.stmts[0]
    executor = GraphExecutor(rules, params)

    x = np.array([[1.0, -2.0], [3.0, 0.5]])
    w0 = np.array([[0.2, -0.3], [0.7, 1.1]])
    params.get("W").value = Value(w0.copy())

    graph = executor.execute(graph_stmt, env={"x": x})
    grad = derive(graph, "y", ["W"], rules)
    grad_w = grad.grads["W"].as_array()

    eps = 1e-5
    fd_w = np.zeros_like(w0)

    def loss_for(w):
        params.get("W").value = Value(w)
        g = executor.execute(graph_stmt, env={"x": x})
        return float(np.sum(g.get("y").value.as_array()))

    for i in range(w0.shape[0]):
        for j in range(w0.shape[1]):
            w_plus = w0.copy()
            w_minus = w0.copy()
            w_plus[i, j] += eps
            w_minus[i, j] -= eps
            fd_w[i, j] = (loss_for(w_plus) - loss_for(w_minus)) / (2 * eps)

    assert np.allclose(grad_w, fd_w, rtol=1e-4, atol=1e-6)
