from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import os
import shutil
import subprocess

import numpy as np

from .. import ast
from .autodiff import Grad, TraceContribution, TraceEntry


class ExplainError(Exception):
    pass


def explain(grad: Grad, level: int = 0) -> str:
    if level < 0 or level > 2:
        raise ExplainError("Explain level must be 0, 1, or 2")
    if level == 0:
        return _format_level0(grad)
    if level == 1:
        return _format_level1(grad)
    return _format_level2(grad)


def render_dot(dot_source: str, output_path: str, fmt: str = "svg") -> None:
    if fmt not in {"svg", "png"}:
        raise ExplainError("render_dot format must be 'svg' or 'png'")
    dot_path = shutil.which("dot")
    if dot_path is None:
        raise ExplainError("Graphviz 'dot' is not available on PATH")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result = subprocess.run(
        [dot_path, f"-T{fmt}", "-o", output_path],
        input=dot_source,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ExplainError(f"Graphviz failed: {result.stderr.strip()}")


def _format_level0(grad: Grad) -> str:
    lines: List[str] = []
    lines.append(f"Gradients for loss '{grad.loss}':")
    lines.append("-" * 45)
    active = _active_node_ids(grad)
    for node in grad.graph.nodes:
        if node.id not in active:
            continue
        value = _fmt_array(node.value.as_array())
        gvalue = _fmt_array(grad.node_grads.get(node.id, np.zeros_like(node.value.as_array())))
        name = node.name or ""
        lines.append(f"[{node.id:02d}] {name:<8} {node.op:<12} value={value} grad={gvalue}")
    return "\n".join(lines)


def _format_level1(grad: Grad) -> str:
    name_map = _build_name_map(grad)
    active = _active_node_ids(grad)
    lines = [_format_level0(grad), "", "digraph G {"]
    for node in grad.graph.nodes:
        if node.id not in active:
            continue
        label = _node_label(node, name_map, grad)
        lines.append(f'  n{node.id} [label="{label}"];')
    for node in grad.graph.nodes:
        if node.id not in active:
            continue
        for src in node.inputs:
            if src in active:
                lines.append(f"  n{src} -> n{node.id};")
    lines.append("}")
    return "\n".join(lines)


def _format_level2(grad: Grad) -> str:
    lines: List[str] = []
    name_map = _build_name_map(grad)
    lines.append(f"Explaining Gradients for loss '{grad.loss}':")
    lines.append("-" * 53)
    for param_name, value in grad.grads.items():
        node_id = _find_param_node_id(grad, param_name)
        if node_id is None:
            continue
        shape = list(value.as_array().shape)
        shape_str = f" (shape {shape})" if shape else ""
        lines.append(f"For {param_name}{shape_str}:")
        paths = _build_paths(grad, node_id, name_map)
        if paths:
            chain_strs = []
            for path in paths:
                chain_strs.append(" * ".join(f"∂{src}/∂{dst}" for src, dst, _, _ in path))
            if len(chain_strs) == 1:
                lines.append(f"∂L/∂{param_name} = {chain_strs[0]}")
            else:
                lines.append(f"∂L/∂{param_name} = " + " + ".join(chain_strs))
        else:
            lines.append(f"∂L/∂{param_name} (no path found)")
        lines.extend(_format_definitions(grad, paths, name_map))
        lines.append("Explanation:")
        lines.extend(_format_trace_for_param(grad, param_name, node_id, name_map))
        lines.extend(_format_local_product(grad, param_name, node_id, paths, name_map))
        lines.append("-" * 53)
    return "\n".join(lines)


def _build_paths(
    grad: Grad, param_node_id: int, name_map: Dict[int, str]
) -> List[List[tuple[str, str, int, int]]]:
    node_grads = grad.node_grads
    if param_node_id not in node_grads:
        return []

    loss_id = grad.graph.name_to_id.get(grad.loss)
    if loss_id is None:
        return []

    trace_map = {entry.node_id: entry for entry in grad.trace}
    paths: List[List[tuple[str, str, int, int]]] = []

    def _walk(node_id: int, current: List[tuple[str, str, int, int]], visited: set[int]) -> None:
        if node_id == param_node_id:
            paths.append(list(current))
            return
        if node_id in visited:
            return
        entry = trace_map.get(node_id)
        if entry is None:
            return
        visited.add(node_id)
        for contrib in entry.contributions:
            src = name_map[node_id]
            dst = name_map[contrib.to_id]
            current.append((src, dst, node_id, contrib.to_id))
            _walk(contrib.to_id, current, visited)
            current.pop()
        visited.remove(node_id)

    _walk(loss_id, [], set())
    return paths


def _format_trace_for_param(
    grad: Grad, param_name: str, param_node_id: int, name_map: Dict[int, str]
) -> List[str]:
    lines: List[str] = []
    lines.append("Trace:")
    for entry in grad.trace:
        targets = {c.to_id for c in entry.contributions}
        if param_node_id not in targets:
            continue
        lines.append(f"  node {name_map[entry.node_id]}")
        lines.append(f"    upstream = {_fmt_array(entry.upstream)}")
        for contrib in entry.contributions:
            if contrib.to_id != param_node_id:
                continue
            lines.append(
                f"    -> {name_map[contrib.to_id]}: "
                f"{contrib.formula} = {_fmt_array(contrib.value)}"
            )
    grad_value = grad.grads.get(param_name)
    if grad_value is not None:
        lines.append(f"  total = {_fmt_array(grad_value.as_array())}")
    return lines


def _format_definitions(
    grad: Grad, paths: List[List[tuple[str, str, int, int]]], name_map: Dict[int, str]
) -> List[str]:
    lines: List[str] = []
    seen: set[int] = set()
    defs: List[str] = []
    for path in paths:
        for _, _, src_id, dst_id in path:
            for node_id in (src_id, dst_id):
                if node_id in seen:
                    continue
                node = grad.graph.nodes[node_id]
                if node.op in {"param", "input"}:
                    continue
                expr_str = _node_expr(node, name_map)
                defs.append(f"    {name_map[node_id]} = {expr_str}")
                seen.add(node_id)
    if defs:
        lines.append("where:")
        lines.extend(defs)
    return lines


def _format_local_product(
    grad: Grad,
    param_name: str,
    param_node_id: int,
    paths: List[List[tuple[str, str, int, int]]],
    name_map: Dict[int, str],
) -> List[str]:
    lines: List[str] = []
    products: List[str] = []
    for path in paths:
        local_terms: List[str] = []
        for _, _, src_id, dst_id in path:
            edge = _find_trace_for_edge(grad.trace, src_id, dst_id)
            if edge is None:
                continue
            entry, contrib = edge
            local_terms.append(_local_term_for_edge(grad, entry, contrib, name_map))
        if local_terms:
            products.append(" * ".join(local_terms))
    if products:
        lines.append("Therefore:")
        if len(products) == 1:
            lines.append(f"    ∂L/∂{param_name} = output_grad * {products[0]}")
        else:
            lines.append(f"    ∂L/∂{param_name} = output_grad * (" + " + ".join(products) + ")")
    return lines


def _find_trace_for_edge(
    trace: List[TraceEntry], src_id: int, dst_id: int
) -> Optional[Tuple[TraceEntry, TraceContribution]]:
    for entry in trace:
        if entry.node_id != src_id:
            continue
        for contrib in entry.contributions:
            if contrib.to_id == dst_id:
                return entry, contrib
    return None


def _find_param_node_id(grad: Grad, param_name: str) -> int | None:
    for node in grad.graph.nodes:
        if node.op == "param" and node.name == param_name:
            return node.id
    return None


def _node_label(node, name_map: Dict[int, str], grad: Grad) -> str:
    shape = list(node.value.as_array().shape)
    shape_str = f"shape={shape}" if shape else "shape=[]"
    value = _fmt_array(node.value.as_array())
    gvalue = _fmt_array(grad.node_grads.get(node.id, np.zeros_like(node.value.as_array())))
    label = (
        f"{node.id}:{name_map[node.id]}:{node.op}\\n"
        f"{shape_str}\\nvalue={value}\\ngrad={gvalue}"
    )
    return label.replace('"', "")


def _node_display(op: str, node_id: int) -> str:
    return f"n{node_id}({op})"


def _node_name_for_chain(grad: Grad, node_id: int) -> str:
    node = grad.graph.nodes[node_id]
    if node.name:
        return node.name
    return _node_display(node.op, node_id)


def _local_term_for_edge(
    grad: Grad, entry: TraceEntry, contrib: TraceContribution, name_map: Dict[int, str]
) -> str:
    node = grad.graph.nodes[entry.node_id]
    op = entry.op
    if op.startswith("rule:"):
        rule_name = op.split(":", 1)[1]
        arg = name_map.get(contrib.to_id, f"n{contrib.to_id}")
        return f"{rule_name}'({arg})"
    if op == "unary:-":
        return "-1"
    if op.startswith("binop:"):
        symbol = op.split(":", 1)[1]
        left_id, right_id = node.inputs[0], node.inputs[1]
        left = name_map[left_id]
        right = name_map[right_id]
        if symbol == "+":
            return "1"
        if symbol == "-":
            return "1" if contrib.to_id == left_id else "-1"
        if symbol == "*":
            return right if contrib.to_id == left_id else left
        if symbol == "/":
            return f"1/{right}" if contrib.to_id == left_id else f"-{left}/({right}^2)"
        if symbol == "@":
            return f"{right}^T" if contrib.to_id == left_id else f"{left}^T"
    if op.startswith("call:"):
        name = op.split(":", 1)[1]
        if name == "exp":
            arg = name_map[node.inputs[0]]
            return f"exp({arg})"
        if name == "max":
            left = name_map[node.inputs[0]]
            right = name_map[node.inputs[1]]
            if contrib.to_id == node.inputs[0]:
                return f"({left} >= {right})"
            return f"({right} > {left})"
    if op == "ternary":
        cond = name_map[node.inputs[0]]
        return f"mask({cond})"
    return contrib.local


def _build_name_map(grad: Grad) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for node in grad.graph.nodes:
        mapping[node.id] = node.name or f"t{node.id}"
    return mapping


def _active_node_ids(grad: Grad) -> set[int]:
    active = set(grad.node_grads.keys())
    if not active:
        loss_id = grad.graph.name_to_id.get(grad.loss)
        if loss_id is not None:
            active.add(loss_id)
    return active


def _node_expr(node, name_map: Dict[int, str]) -> str:
    if node.expr:
        return _expr_to_str(node.expr)
    if node.op == "const":
        return _fmt_array(node.value.as_array())
    if node.op == "list":
        return _fmt_array(node.value.as_array())
    if node.op.startswith("binop:"):
        op = node.op.split(":", 1)[1]
        left = name_map[node.inputs[0]]
        right = name_map[node.inputs[1]]
        return f"{left} {op} {right}"
    if node.op == "unary:-":
        return f"-{name_map[node.inputs[0]]}"
    if node.op == "ternary":
        cond = name_map[node.inputs[0]]
        then_expr = name_map[node.inputs[1]]
        else_expr = name_map[node.inputs[2]]
        return f"({cond} ? {then_expr} : {else_expr})"
    if node.op.startswith("call:"):
        fn = node.op.split(":", 1)[1]
        args = ", ".join(name_map[idx] for idx in node.inputs)
        return f"{fn}({args})"
    if node.op.startswith("rule:"):
        fn = node.op.split(":", 1)[1]
        args = ", ".join(name_map[idx] for idx in node.inputs)
        return f"{fn}({args})"
    return node.op


def _expr_to_str(expr: ast.Expr) -> str:
    if isinstance(expr, ast.Number):
        return str(expr.value)
    if isinstance(expr, ast.StringLiteral):
        return f"\"{expr.value}\""
    if isinstance(expr, ast.ListLiteral):
        items = ", ".join(_expr_to_str(item) for item in expr.items)
        return f"[{items}]"
    if isinstance(expr, ast.IndexExpr):
        return f"{_expr_to_str(expr.base)}[{_expr_to_str(expr.index)}]"
    if isinstance(expr, ast.Name):
        return expr.value
    if isinstance(expr, ast.UnaryOp):
        return f"-{_expr_to_str(expr.expr)}"
    if isinstance(expr, ast.BinaryOp):
        left = _expr_to_str(expr.left)
        right = _expr_to_str(expr.right)
        return f"{left} {expr.op} {right}"
    if isinstance(expr, ast.TernaryOp):
        cond = _expr_to_str(expr.cond)
        then_expr = _expr_to_str(expr.then_expr)
        else_expr = _expr_to_str(expr.else_expr)
        return f"({cond} ? {then_expr} : {else_expr})"
    if isinstance(expr, ast.CallExpr):
        args = []
        for arg in expr.args:
            value = _expr_to_str(arg.value)
            if arg.name:
                args.append(f"{arg.name}={value}")
            else:
                args.append(value)
        arg_str = ", ".join(args)
        return f"{expr.func}({arg_str})"
    return "<expr>"


def _fmt_array(value: np.ndarray) -> str:
    if np.isscalar(value) or value.shape == ():
        return str(float(value))
    return np.array2string(value, precision=4, separator=",")
