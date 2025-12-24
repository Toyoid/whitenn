from whitenn import ast
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
