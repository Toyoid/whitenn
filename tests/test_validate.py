import pytest

from whitenn.parser import parse_program
from whitenn.validate import ValidationError, validate_program


def test_validate_ok():
    source = """
    model M {
      param W[2,3] init normal(0, 0.1);
    }

    rule relu(x: Real) : Real {
      forward = max(0, x);
      d/dx = (x > 0) ? 1 : 0;
    }

    fn train_step(x[2]) {
      graph {
        y = relu(x @ W);
      }
      grad g = derive y wrt {W};
      step SGD(lr=0.1) using g;
    }
    """
    program = parse_program(source)
    validate_program(program)


def test_missing_forward_in_rule():
    source = """
    rule r(x: Real) : Real {
      d/dx = 1;
    }
    """
    with pytest.raises(ValidationError) as excinfo:
        validate_program(parse_program(source))
    assert "Missing forward clause" in str(excinfo.value)


def test_missing_derivative_in_rule():
    source = """
    rule r(x: Real, y: Real) : Real {
      forward = x + y;
      d/dx = 1;
    }
    """
    with pytest.raises(ValidationError) as excinfo:
        validate_program(parse_program(source))
    assert "Missing derivative clause" in str(excinfo.value)


def test_nondiff_derivative_error():
    source = """
    rule r(x: Real nondiff) : Real {
      forward = x;
      d/dx = 1;
    }
    """
    with pytest.raises(ValidationError) as excinfo:
        validate_program(parse_program(source))
    assert "nondiff" in str(excinfo.value)


def test_duplicate_top_level_name():
    source = """
    rule foo(x: Real) : Real {
      forward = x;
      d/dx = 1;
    }

    fn foo() {
      graph { y = 1; }
    }
    """
    with pytest.raises(ValidationError) as excinfo:
        validate_program(parse_program(source))
    assert "Duplicate top-level name" in str(excinfo.value)
