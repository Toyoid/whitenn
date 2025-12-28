import shutil

import pytest
import numpy as np

from whitenn.interpreter import Interpreter
from whitenn.parser import parse_program


def test_interpreter_train_step_updates_param():
    source = """
    model M {
      param W init zeros;
    }

    fn train_step(x) {
      graph {
        L = x * W;
      }
      loss Loss = L;
      grad g = derive Loss wrt {W};
      step SGD(lr=0.1) using g;
    }
    """
    program = parse_program(source)
    interpreter = Interpreter(program)
    interpreter.run_function("train_step", {"x": 2.0})
    param = interpreter.params.get("W").value.as_array()
    assert float(param) == -0.2


def test_interpreter_loss_expression_creates_node():
    source = """
    model M {
      param W init zeros;
    }

    fn train_step(x) {
      graph {
        y = x * W;
      }
      loss L = y + y;
      grad g = derive L wrt {W};
    }
    """
    program = parse_program(source)
    interpreter = Interpreter(program)
    ctx = interpreter.run_function("train_step", {"x": 2.0})
    assert ctx.last_graph is not None
    loss_node = ctx.last_graph.get("L")
    assert loss_node.op == "binop:+"


def test_interpreter_loss_rejects_host_name():
    source = """
    model M {
      param W init zeros;
    }

    fn train_step(x) {
      graph {
        y = x * W;
      }
      loss L = y + outside;
      grad g = derive L wrt {W};
    }
    """
    program = parse_program(source)
    try:
        Interpreter(program)
    except Exception as exc:
        assert "loss references non-graph name" in str(exc)
    else:
        raise AssertionError("Expected loss validation error")


def test_interpreter_explain_output():
    source = """
    model M {
      param W init zeros;
      param b init zeros;
    }

    rule relu(x: Real) : Real {
      forward = max(0, x);
      d/dx = (x > 0) ? 1 : 0;
    }

    fn train_step(x) {
      graph {
        y = relu(x * W + b);
      }
      grad g = derive y wrt {W, b};
      explain g level 0;
    }
    """
    program = parse_program(source)
    interpreter = Interpreter(program)
    ctx = interpreter.run_function("train_step", {"x": 2.0})
    assert ctx.last_explain is not None
    assert "Gradients for loss" in ctx.last_explain
    assert ctx.explain_outputs


def test_interpreter_explain_to_svg(tmp_path):
    if shutil.which("dot") is None:
        pytest.skip("graphviz dot not installed")
    source = """
    model M {
      param W init zeros;
      param b init zeros;
    }

    rule relu(x: Real) : Real {
      forward = max(0, x);
      d/dx = (x > 0) ? 1 : 0;
    }

    fn train_step(x, out_path) {
      graph {
        y = relu(x * W + b);
      }
      grad g = derive y wrt {W, b};
      explain g level 1 to out_path;
    }
    """
    out_path = tmp_path / "graph.svg"
    program = parse_program(source.replace("OUTPATH", str(out_path)))
    interpreter = Interpreter(program)
    ctx = interpreter.run_function("train_step", {"x": 2.0, "out_path": str(out_path)})
    assert ctx.last_explain is not None
    assert not ctx.explain_outputs
    assert out_path.exists()


def test_interpreter_for_loop_executes():
    source = """
    model M {
      param W init zeros;
    }

    fn train_step(x) {
      graph {
        L = x * W;
      }
      grad g = derive L wrt {W};
      step SGD(lr=0.1) using g;
    }

    fn train_epochs() {
      for epoch in 1..3 {
        train_step(1);
      }
    }
    """
    program = parse_program(source)
    interpreter = Interpreter(program)
    interpreter.run_function("train_epochs", {})
    param = interpreter.params.get("W").value.as_array()
    # Each step: W := W - 0.1 * x, with x=1 and starting at 0.
    assert float(param) == pytest.approx(-0.3)


def test_interpreter_function_call_assignment():
    source = """
    model M {
      param W init zeros;
    }

    fn train_step(x) {
      graph {
        L = x * W;
      }
      grad g = derive L wrt {W};
      step SGD(lr=0.2) using g;
    }

    fn run() {
      train_step(2);
      train_step(2);
    }
    """
    program = parse_program(source)
    interpreter = Interpreter(program)
    interpreter.run_function("run", {})
    param = interpreter.params.get("W").value.as_array()
    # Each step: W := W - 0.2 * x, with x=2 and starting at 0.
    assert float(param) == -0.8


def test_interpreter_list_literal_args():
    source = """
    model M {
      param W[2] init zeros;
    }

    fn train_step(x) {
      graph {
        y = x + W;
      }
      loss L = y + y;
      grad g = derive L wrt {W};
      step SGD(lr=0.1) using g;
    }

    train_step([1, 2]);
    """
    program = parse_program(source)
    interpreter = Interpreter(program)
    interpreter.run_program()
    param = interpreter.params.get("W").value.as_array()
    assert np.allclose(param, np.array([-0.2, -0.2]))


def test_interpreter_fetch_graph_value():
    source = """
    model M {
      param W init zeros;
    }

    fn f() {
      graph {
        y = W + 1;
      }
      fetch y_val = y;
    }
    """
    program = parse_program(source)
    interpreter = Interpreter(program)
    ctx = interpreter.run_function("f", {})
    assert ctx.env["y_val"] == 1.0


def test_interpreter_indexing_and_builtins():
    source = """
    fn f() {
      data = [1, 2, 3];
      x = data[1];
      grid = linspace(0, 1, 3);
      n = len(grid);
      noise = randn(2, 2);
      z = zeros(2, 1);
      o = ones(1, 2);
    }
    """
    program = parse_program(source)
    interpreter = Interpreter(program)
    ctx = interpreter.run_function("f", {})
    assert ctx.env["x"] == 2
    assert np.allclose(ctx.env["grid"], np.array([0.0, 0.5, 1.0]))
    assert ctx.env["n"] == 3
    assert ctx.env["noise"].shape == (2, 2)
    assert np.allclose(ctx.env["z"], np.zeros((2, 1)))
    assert np.allclose(ctx.env["o"], np.ones((1, 2)))


def test_interpreter_argsort_one_hot():
    source = """
    fn f() {
      x = [3, 1, 2];
      idx = argsort(x);
      oh = one_hot(idx, 3);
    }
    """
    program = parse_program(source)
    interpreter = Interpreter(program)
    ctx = interpreter.run_function("f", {})
    assert np.allclose(ctx.env["idx"], np.array([1, 2, 0]))
    expected = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    assert np.allclose(ctx.env["oh"], expected)


def test_interpreter_argmax_accuracy():
    source = """
    fn accuracy(preds, target) {
      pred_idx = argmax(preds);
      tgt_idx = argmax(target);
      eq = pred_idx == tgt_idx;
      return mean(eq);
    }

    fn f() {
      probs = [[0.1, 0.9], [0.8, 0.2]];
      target = [[0, 1], [1, 0]];
      acc = accuracy(probs, target);
    }
    """
    program = parse_program(source)
    interpreter = Interpreter(program)
    ctx = interpreter.run_function("f", {})
    assert ctx.env["acc"] == 1.0


def test_interpreter_return_value():
    source = """
    fn f() {
      return 3;
    }
    """
    program = parse_program(source)
    interpreter = Interpreter(program)
    ctx = interpreter.run_function("f", {})
    assert ctx.return_value == 3.0


def test_interpreter_return_value_from_call():
    source = """
    fn inner() {
      return 5;
    }

    fn outer() {
      x = inner();
      return x;
    }
    """
    program = parse_program(source)
    interpreter = Interpreter(program)
    ctx = interpreter.run_function("outer", {})
    assert ctx.return_value == 5.0
