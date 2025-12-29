import pytest

from whitenn.interpreter import Interpreter
from whitenn.parser import parse_program


def _run_and_capture_error(source: str, filename: str = "bad.wnn") -> str:
    program = parse_program(source, filename=filename)
    interpreter = Interpreter(program)
    with pytest.raises(Exception) as excinfo:
        interpreter.run_program()
    return str(excinfo.value)


def test_error_location_unknown_name():
    source = """
    fn f() {
      x = y + 1;
    }

    f();
    """
    message = _run_and_capture_error(source)
    assert "Unknown name 'y'" in message
    assert "bad.wnn" in message
    assert "^" in message


def test_error_location_explain_level1_requires_path():
    source = """
    model M {
      param W init zeros;
    }

    fn f() {
      graph { y = W + 1; }
      grad g = derive y wrt {W};
      explain g level 1;
    }

    f();
    """
    message = _run_and_capture_error(source)
    assert "Explain level 1 requires an output path" in message
    assert "bad.wnn" in message
    assert "^" in message


def test_error_location_for_loop_bounds():
    source = """
    fn f() {
      for i in "a"..2 {
        x = i;
      }
    }

    f();
    """
    message = _run_and_capture_error(source)
    assert "for loop start must be an integer" in message
    assert "bad.wnn" in message
    assert "^" in message


def test_error_location_unknown_rule_in_graph():
    source = """
    fn f(x) {
      graph { y = foo(x); }
    }

    f(1);
    """
    message = _run_and_capture_error(source)
    assert "Unknown rule 'foo'" in message
    assert "bad.wnn" in message
    assert "^" in message


def test_error_location_optimizer_lr():
    source = """
    model M {
      param W init zeros;
    }

    fn f() {
      graph { y = W + 1; }
      grad g = derive y wrt {W};
      step SGD(lr="bad") using g;
    }

    f();
    """
    message = _run_and_capture_error(source)
    assert "SGD lr must be numeric" in message
    assert "bad.wnn" in message
    assert "^" in message


def test_error_location_if_condition_scalar():
    source = """
    fn f() {
      if [1, 2] {
        x = 1;
      }
    }

    f();
    """
    message = _run_and_capture_error(source)
    assert "if condition must be a scalar" in message
    assert "bad.wnn" in message
    assert "^" in message


def test_error_location_boolean_operator_scalar():
    source = """
    fn f() {
      if [1, 2] && 1 {
        x = 1;
      }
    }

    f();
    """
    message = _run_and_capture_error(source)
    assert "if condition must be a scalar" in message
    assert "bad.wnn" in message
    assert "^" in message
