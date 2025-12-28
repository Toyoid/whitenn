import pytest

from whitenn.parser import parse_program
from whitenn.runtime.rules import RuleError, RuleTable


def test_rule_table_from_program():
    source = """
    rule relu(x: Real) : Real {
      forward = max(0, x);
      d/dx = (x > 0) ? 1 : 0;
    }
    """
    program = parse_program(source)
    table = RuleTable.from_program(program)

    rule = table.get("relu")
    assert rule.name == "relu"
    assert rule.param_names == ["x"]
    assert "x" in rule.derivs


def test_rule_missing_forward():
    source = """
    rule bad(x: Real) : Real {
      d/dx = 1;
    }
    """
    program = parse_program(source)
    with pytest.raises(RuleError):
        RuleTable.from_program(program)


def test_rule_duplicate_name():
    source = """
    rule r(x: Real) : Real {
      forward = x;
      d/dx = 1;
    }

    rule r(y: Real) : Real {
      forward = y;
      d/dy = 1;
    }
    """
    program = parse_program(source)
    with pytest.raises(RuleError):
        RuleTable.from_program(program)
