import numpy as np

from whitenn.parser import parse_program
from whitenn.runtime.graph import GraphExecutor
from whitenn.runtime.params import ParamStore, init_zeros
from whitenn.runtime.rules import RuleTable
from whitenn.runtime.values import Value


def test_graph_exec_simple_rule():
    source = """
    model M {
      param W[2] init zeros;
    }

    rule add(x: Real, y: Real) : Real {
      forward = x + y;
      d/dx = 1;
      d/dy = 1;
    }

    fn f(x[2]) {
      graph {
        y = add(x, W);
      }
    }
    """
    program = parse_program(source)
    rules = RuleTable.from_program(program)
    params = ParamStore(seed=0)
    params.add_param("W", [2], init=init_zeros)
    params.get("W").value = Value(np.array([1.0, 2.0]))

    graph_stmt = program.items[2].body.stmts[0]
    executor = GraphExecutor(rules, params)
    graph = executor.execute(graph_stmt, env={"x": np.array([3.0, 4.0])})

    node = graph.get("y")
    assert np.allclose(node.value.as_array(), np.array([4.0, 6.0]))


def test_graph_exec_nested_ops():
    source = """
    model M {
      param W1[2,2] init zeros;
      param b1[2] init zeros;
    }

    rule relu(x: Real) : Real {
      forward = max(0, x);
      d/dx = (x > 0) ? 1 : 0;
    }

    fn f(x[2,2]) {
      graph {
        h = relu(x @ W1 + b1);
      }
    }
    """
    program = parse_program(source)
    rules = RuleTable.from_program(program)
    params = ParamStore(seed=0)
    params.add_param("W1", [2, 2], init=init_zeros)
    params.add_param("b1", [2], init=init_zeros)
    params.get("W1").value = Value(np.array([[1.0, 0.0], [0.0, 1.0]]))
    params.get("b1").value = Value(np.array([1.0, 1.0]))

    graph_stmt = program.items[2].body.stmts[0]
    executor = GraphExecutor(rules, params)
    x = np.array([[1.0, -2.0], [3.0, 0.0]])
    graph = executor.execute(graph_stmt, env={"x": x})

    h_node = graph.get("h")
    assert h_node.op == "rule:relu"

    plus_id = h_node.inputs[0]
    plus_node = graph.nodes[plus_id]
    assert plus_node.op == "binop:+"
    plus_input_ops = {graph.nodes[node_id].op for node_id in plus_node.inputs}
    assert "binop:@" in plus_input_ops
    assert "param" in plus_input_ops

    matmul_id = next(node_id for node_id in plus_node.inputs if graph.nodes[node_id].op == "binop:@")
    matmul_ops = {graph.nodes[node_id].op for node_id in graph.nodes[matmul_id].inputs}
    assert "input" in matmul_ops
    assert "param" in matmul_ops

    expected = np.array([[2.0, 0.0], [4.0, 1.0]])
    assert np.allclose(h_node.value.as_array(), expected)


def test_graph_exec_softmax_and_transpose():
    source = """
    fn f() {
      graph {
        y = softmax(x);
        z = transpose(W);
      }
    }
    """
    program = parse_program(source)
    rules = RuleTable.from_program(program)
    params = ParamStore(seed=0)
    params.add_param("W", [2, 3], init=init_zeros)
    params.get("W").value = Value(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    graph_stmt = program.items[0].body.stmts[0]
    executor = GraphExecutor(rules, params)
    x = np.array([1.0, 2.0, 3.0])
    graph = executor.execute(graph_stmt, env={"x": x})

    y_node = graph.get("y")
    z_node = graph.get("z")
    assert np.allclose(y_node.value.as_array(), np.exp(x) / np.sum(np.exp(x)))
    assert np.allclose(z_node.value.as_array(), params.get("W").value.as_array().T)
