import pytest

from whitenn import ast
from whitenn.errors import ParseError
from whitenn.parser import parse_program


def test_parse_minimal_program():
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
      explain g level 1;
      step SGD(lr=0.1) using g;
    }
    """
    program = parse_program(source)

    assert isinstance(program, ast.Program)
    assert len(program.items) == 3
    assert isinstance(program.items[0], ast.ModelDecl)
    assert isinstance(program.items[1], ast.RuleDecl)
    assert isinstance(program.items[2], ast.FnDecl)


def test_parse_design_example_snippet():
    source = """
    model MLP {
      param W1[2,3] init normal(0, 0.1);
      param b1[3]   init zeros;
      param W2[3,1] init normal(0, 0.1);
      param b2[1]   init zeros;
    }

    rule relu(x: Real) : Real {
      forward = max(0, x);
      d/dx    = (x > 0) ? 1 : 0;
    }

    rule sigmoid(x: Real) : Real {
      forward = 1 / (1 + exp(-x));
      d/dx    = sigmoid(x) * (1 - sigmoid(x));
    }

    fn train_step(x[2], t[1]) {
      graph {
        h = relu(x @ W1 + b1);
        y = sigmoid(h @ W2 + b2);
        L = mse(y, t);
      }

      grad g = derive L wrt {W1, b1, W2, b2};
      explain g level 2;
      step SGD(lr=0.1) using g;
    }

    for epoch in 1..1000 {
      train_step(sample_x, sample_t);
    }
    """
    program = parse_program(source)

    assert isinstance(program, ast.Program)
    assert len(program.items) == 5
    assert isinstance(program.items[0], ast.ModelDecl)
    assert isinstance(program.items[1], ast.RuleDecl)
    assert isinstance(program.items[2], ast.RuleDecl)
    assert isinstance(program.items[3], ast.FnDecl)
    assert isinstance(program.items[4], ast.ForStmt)


def test_parse_loss_statement():
    source = """
    fn f() {
      graph {
        L = 1;
      }
      loss L1 = L;
      grad g = derive L1 wrt {W};
    }
    """
    program = parse_program(source)
    fn = program.items[0]
    assert isinstance(fn, ast.FnDecl)
    assert any(isinstance(stmt, ast.LossStmt) for stmt in fn.body.stmts)


def test_parse_loss_expression():
    source = """
    fn f() {
      graph {
        y = 1;
      }
      loss L = y + y;
    }
    """
    program = parse_program(source)
    fn = program.items[0]
    loss_stmt = next(stmt for stmt in fn.body.stmts if isinstance(stmt, ast.LossStmt))
    assert isinstance(loss_stmt.expr, ast.BinaryOp)


def test_parse_if_statement():
    source = """
    fn f(x) {
      if x > 0 && x < 3 {
        y = 1;
      } else {
        y = -1;
      }
    }
    """
    program = parse_program(source)
    fn = program.items[0]
    if_stmt = next(stmt for stmt in fn.body.stmts if isinstance(stmt, ast.IfStmt))
    assert isinstance(if_stmt.cond, ast.BinaryOp)
    assert if_stmt.else_block is not None


def test_parse_list_literal():
    source = """
    fn f() {
      graph {
        y = [[1, 2], [3, 4]];
      }
    }
    """
    program = parse_program(source)
    fn = program.items[0]
    graph_stmt = next(stmt for stmt in fn.body.stmts if isinstance(stmt, ast.GraphStmt))
    item = graph_stmt.items[0]
    assert isinstance(item.expr, ast.ListLiteral)
    assert isinstance(item.expr.items[0], ast.ListLiteral)


def test_parse_string_literal():
    source = """
    fn f() {
      print("hello");
    }
    """
    program = parse_program(source)
    fn = program.items[0]
    expr_stmt = next(stmt for stmt in fn.body.stmts if isinstance(stmt, ast.ExprStmt))
    assert isinstance(expr_stmt.expr, ast.CallExpr)
    assert isinstance(expr_stmt.expr.args[0].value, ast.StringLiteral)


def test_parse_fetch_statement():
    source = """
    fn f() {
      graph {
        y = 1;
      }
      fetch y_val = y;
    }
    """
    program = parse_program(source)
    fn = program.items[0]
    fetch_stmt = next(stmt for stmt in fn.body.stmts if isinstance(stmt, ast.FetchStmt))
    assert fetch_stmt.target_name == "y_val"
    assert fetch_stmt.source_name == "y"


def test_parse_index_expr():
    source = """
    fn f() {
      x = data[0];
    }
    """
    program = parse_program(source)
    fn = program.items[0]
    assign_stmt = next(stmt for stmt in fn.body.stmts if isinstance(stmt, ast.AssignStmt))
    assert isinstance(assign_stmt.expr, ast.IndexExpr)


def test_parse_log_call():
    source = """
    fn f() {
      graph { y = log(x); }
    }
    """
    program = parse_program(source)
    fn = program.items[0]
    graph_stmt = next(stmt for stmt in fn.body.stmts if isinstance(stmt, ast.GraphStmt))
    call = graph_stmt.items[0].expr
    assert isinstance(call, ast.CallExpr)
    assert call.func == "log"


def test_parse_return_statement():
    source = """
    fn f() {
      return 1;
    }
    """
    program = parse_program(source)
    fn = program.items[0]
    stmt = fn.body.stmts[0]
    assert isinstance(stmt, ast.ReturnStmt)


def test_parse_error_has_location():
    source = "fn f() { graph { y = ; } }"
    with pytest.raises(ParseError) as excinfo:
        parse_program(source, filename="bad.wnn")
    message = str(excinfo.value)
    assert "bad.wnn" in message
    assert "^" in message
