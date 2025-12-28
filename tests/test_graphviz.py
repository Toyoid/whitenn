import os
import shutil
import tempfile

import pytest

from whitenn.runtime.explain import ExplainError, render_dot


def test_render_dot_svg(tmp_path):
    if shutil.which("dot") is None:
        pytest.skip("graphviz dot not installed")
    dot_source = "digraph G { a -> b; }"
    out_path = tmp_path / "graph.svg"
    render_dot(dot_source, str(out_path), fmt="svg")
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_render_dot_invalid_format():
    with pytest.raises(ExplainError):
        render_dot("digraph G { a -> b; }", "out.svg", fmt="pdf")
