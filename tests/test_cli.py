from pathlib import Path

from whitenn.cli import main


def test_cli_runs_and_prints_explain(tmp_path, capsys):
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
      loss L = y + y;
      grad g = derive L wrt {W, b};
      explain g level 1;
    }

    train_step(2);
    """
    file_path = tmp_path / "prog.wnn"
    file_path.write_text(source, encoding="utf-8")

    exit_code = main([str(file_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "digraph G" in captured.out


def test_cli_rejects_non_wnn(tmp_path, capsys):
    file_path = tmp_path / "prog.txt"
    file_path.write_text("fn f() { graph { y = 1; } }", encoding="utf-8")

    exit_code = main([str(file_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Expected a .wnn file" in captured.err
