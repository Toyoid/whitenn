import numpy as np

from whitenn.parser import parse_program
from whitenn.runtime.autodiff import derive
from whitenn.runtime.explain import explain
from whitenn.runtime.graph import GraphExecutor
from whitenn.runtime.params import ParamStore, init_zeros
from whitenn.runtime.rules import RuleTable
from whitenn.runtime.values import Value


def _build_simple_graph():
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
    return rules, graph


def test_explain_level0_includes_nodes():
    rules, graph = _build_simple_graph()
    grad = derive(graph, "y", ["W", "b"], rules)
    text = explain(grad, level=0)
    assert "Gradients for loss 'y'" in text
    assert "rule:relu" in text
    assert "binop:*" in text


def test_explain_level1_includes_dot():
    rules, graph = _build_simple_graph()
    grad = derive(graph, "y", ["W", "b"], rules)
    text = explain(grad, level=1)
    assert "digraph G" in text
    assert "->" in text
    assert "grad=" in text


def test_explain_level2_includes_trace():
    rules, graph = _build_simple_graph()
    grad = derive(graph, "y", ["W", "b"], rules)
    text = explain(grad, level=2)
    assert "Explaining Gradients" in text
    assert "Trace:" in text


def test_explain_level2_multiple_paths():
    source = """
    model M {
      param W init zeros;
    }

    fn f() {
      graph {
        y = x * W + x * W;
      }
    }
    """
    program = parse_program(source)
    rules = RuleTable.from_program(program)
    params = ParamStore(seed=0)
    params.add_param("W", None, init=init_zeros)
    params.get("W").value = Value(2.0)
    graph_stmt = program.items[1].body.stmts[0]
    graph = GraphExecutor(rules, params).execute(graph_stmt, env={"x": 3.0})
    grad = derive(graph, "y", ["W"], rules)
    text = explain(grad, level=2)
    chain_line = next(line for line in text.splitlines() if line.startswith("∂L/∂W = "))
    assert " + " in chain_line
