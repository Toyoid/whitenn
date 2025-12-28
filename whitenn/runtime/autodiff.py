from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .. import ast
from .graph import Graph, GraphError, _eval_expr_value
from ..errors import RuntimeError, format_error
from .rules import RuleTable
from .values import Value


class AutodiffError(RuntimeError):
    pass


@dataclass(frozen=True)
class Grad:
    grads: Dict[str, Value]
    loss: str
    graph: Graph
    node_grads: Dict[int, np.ndarray]
    trace: List["TraceEntry"]


@dataclass(frozen=True)
class TraceContribution:
    to_id: int
    formula: str
    local: str
    value: np.ndarray


@dataclass(frozen=True)
class TraceEntry:
    node_id: int
    op: str
    upstream: np.ndarray
    contributions: List[TraceContribution]


def derive(
    graph: Graph,
    loss_name: str,
    params: Sequence[str],
    rules: RuleTable,
    source: str | None = None,
    filename: str | None = None,
) -> Grad:
    loss_id = graph.name_to_id.get(loss_name)
    if loss_id is None:
        raise AutodiffError(f"Unknown loss node '{loss_name}'")

    node_grads: Dict[int, np.ndarray] = {}
    trace: List[TraceEntry] = []
    loss_value = graph.nodes[loss_id].value.as_array()
    node_grads[loss_id] = np.ones_like(loss_value)

    for node in reversed(graph.nodes):
        if node.id not in node_grads:
            continue
        upstream = node_grads[node.id]
        try:
            entry = _backprop_node(node, upstream, node_grads, graph, rules)
        except AutodiffError as exc:
            span = getattr(node.expr, "span", None)
            message = format_error(
                str(exc), span=span, source=source, filename=filename
            )
            raise AutodiffError(message) from exc
        if entry is not None:
            trace.append(entry)

    param_nodes = _param_name_to_id(graph)
    grads: Dict[str, Value] = {}
    for name in params:
        if name not in param_nodes:
            raise AutodiffError(f"Unknown parameter '{name}' in grad list")
        node_id = param_nodes[name]
        grad_value = node_grads.get(node_id, np.zeros_like(graph.nodes[node_id].value.as_array()))
        grads[name] = Value(grad_value)

    return Grad(grads=grads, loss=loss_name, graph=graph, node_grads=node_grads, trace=trace)


def _backprop_node(
    node, upstream: np.ndarray, node_grads: Dict[int, np.ndarray], graph: Graph, rules: RuleTable
) -> TraceEntry | None:
    if node.op in {"const", "input", "param", "list"}:
        return None
    if node.op == "unary:-":
        contribution = TraceContribution(
            to_id=node.inputs[0], formula="upstream * (-1)", local="-1", value=-upstream
        )
        _add_grad(node_grads, node.inputs[0], contribution.value)
        return TraceEntry(node.id, node.op, upstream, [contribution])
    if node.op.startswith("binop:"):
        return _backprop_binop(node, upstream, node_grads, graph)
    if node.op == "ternary":
        cond = graph.nodes[node.inputs[0]].value.as_array()
        mask = cond.astype(bool)
        left_value = upstream * mask
        right_value = upstream * (~mask)
        contributions = [
            TraceContribution(node.inputs[1], "upstream * mask", "mask", left_value),
            TraceContribution(node.inputs[2], "upstream * ~mask", "~mask", right_value),
        ]
        _add_grad(node_grads, node.inputs[1], left_value)
        _add_grad(node_grads, node.inputs[2], right_value)
        return TraceEntry(node.id, node.op, upstream, contributions)
    if node.op.startswith("call:"):
        return _backprop_builtin(node, upstream, node_grads, graph)
    if node.op.startswith("rule:"):
        return _backprop_rule(node, upstream, node_grads, graph, rules)
    raise AutodiffError(f"Unsupported op '{node.op}' in autodiff")


def _backprop_binop(node, upstream, node_grads, graph: Graph) -> TraceEntry:
    op = node.op.split(":", 1)[1]
    left = graph.nodes[node.inputs[0]].value.as_array()
    right = graph.nodes[node.inputs[1]].value.as_array()

    if op == "+":
        left_contrib = _reduce_grad(upstream, left.shape)
        right_contrib = _reduce_grad(upstream, right.shape)
        contributions = [
            TraceContribution(node.inputs[0], "upstream * 1", "1", left_contrib),
            TraceContribution(node.inputs[1], "upstream * 1", "1", right_contrib),
        ]
        _add_grad(node_grads, node.inputs[0], left_contrib)
        _add_grad(node_grads, node.inputs[1], right_contrib)
        return TraceEntry(node.id, node.op, upstream, contributions)
    if op == "-":
        left_contrib = _reduce_grad(upstream, left.shape)
        right_contrib = _reduce_grad(-upstream, right.shape)
        contributions = [
            TraceContribution(node.inputs[0], "upstream * 1", "1", left_contrib),
            TraceContribution(node.inputs[1], "upstream * -1", "-1", right_contrib),
        ]
        _add_grad(node_grads, node.inputs[0], left_contrib)
        _add_grad(node_grads, node.inputs[1], right_contrib)
        return TraceEntry(node.id, node.op, upstream, contributions)
    if op == "*":
        left_contrib = _reduce_grad(upstream * right, left.shape)
        right_contrib = _reduce_grad(upstream * left, right.shape)
        contributions = [
            TraceContribution(node.inputs[0], "upstream * right", "right", left_contrib),
            TraceContribution(node.inputs[1], "upstream * left", "left", right_contrib),
        ]
        _add_grad(node_grads, node.inputs[0], left_contrib)
        _add_grad(node_grads, node.inputs[1], right_contrib)
        return TraceEntry(node.id, node.op, upstream, contributions)
    if op == "/":
        left_contrib = _reduce_grad(upstream / right, left.shape)
        right_contrib = _reduce_grad(-upstream * left / (right ** 2), right.shape)
        contributions = [
            TraceContribution(node.inputs[0], "upstream / right", "1/right", left_contrib),
            TraceContribution(
                node.inputs[1], "-upstream * left / (right ** 2)", "-left/(right**2)", right_contrib
            ),
        ]
        _add_grad(node_grads, node.inputs[0], left_contrib)
        _add_grad(node_grads, node.inputs[1], right_contrib)
        return TraceEntry(node.id, node.op, upstream, contributions)
    if op == "@":
        left_contrib, right_contrib = _matmul_grads(upstream, left, right)
        contributions = [
            TraceContribution(node.inputs[0], "upstream @ right.T", "right.T", left_contrib),
            TraceContribution(node.inputs[1], "left.T @ upstream", "left.T", right_contrib),
        ]
        _add_grad(node_grads, node.inputs[0], left_contrib)
        _add_grad(node_grads, node.inputs[1], right_contrib)
        return TraceEntry(node.id, node.op, upstream, contributions)
    raise AutodiffError(f"Unsupported binary op '{op}' in autodiff")


def _backprop_builtin(node, upstream, node_grads, graph: Graph) -> TraceEntry:
    name = node.op.split(":", 1)[1]
    if name == "exp":
        x = graph.nodes[node.inputs[0]].value.as_array()
        contrib = upstream * np.exp(x)
        _add_grad(node_grads, node.inputs[0], contrib)
        return TraceEntry(
            node.id,
            node.op,
            upstream,
            [TraceContribution(node.inputs[0], "upstream * exp(x)", "exp(x)", contrib)],
        )
    if name == "log":
        x = graph.nodes[node.inputs[0]].value.as_array()
        contrib = upstream / x
        _add_grad(node_grads, node.inputs[0], contrib)
        return TraceEntry(
            node.id,
            node.op,
            upstream,
            [TraceContribution(node.inputs[0], "upstream / x", "1/x", contrib)],
        )
    if name == "sum":
        x = graph.nodes[node.inputs[0]].value.as_array()
        contrib = np.ones_like(x) * upstream
        _add_grad(node_grads, node.inputs[0], contrib)
        return TraceEntry(
            node.id,
            node.op,
            upstream,
            [TraceContribution(node.inputs[0], "upstream * 1", "1", contrib)],
        )
    if name == "mean":
        x = graph.nodes[node.inputs[0]].value.as_array()
        scale = 0.0 if x.size == 0 else 1.0 / x.size
        contrib = np.ones_like(x) * upstream * scale
        _add_grad(node_grads, node.inputs[0], contrib)
        return TraceEntry(
            node.id,
            node.op,
            upstream,
            [
                TraceContribution(
                    node.inputs[0], "upstream * (1/n)", "1/n", contrib
                )
            ],
        )
    if name == "transpose":
        x = graph.nodes[node.inputs[0]].value.as_array()
        if x.ndim < 2:
            contrib = upstream
        else:
            contrib = np.swapaxes(upstream, -1, -2)
        _add_grad(node_grads, node.inputs[0], contrib)
        return TraceEntry(
            node.id,
            node.op,
            upstream,
            [
                TraceContribution(
                    node.inputs[0], "upstream.T", "transpose", contrib
                )
            ],
        )
    if name == "softmax":
        s = node.value.as_array()
        dot = np.sum(upstream * s, axis=-1, keepdims=True)
        contrib = s * (upstream - dot)
        _add_grad(node_grads, node.inputs[0], contrib)
        return TraceEntry(
            node.id,
            node.op,
            upstream,
            [
                TraceContribution(
                    node.inputs[0], "upstream * softmax_jacobian", "softmax_jacobian", contrib
                )
            ],
        )
    if name == "max":
        a = graph.nodes[node.inputs[0]].value.as_array()
        b = graph.nodes[node.inputs[1]].value.as_array()
        mask_a = a >= b
        mask_b = b > a
        left_contrib = upstream * mask_a
        right_contrib = upstream * mask_b
        _add_grad(node_grads, node.inputs[0], left_contrib)
        _add_grad(node_grads, node.inputs[1], right_contrib)
        return TraceEntry(
            node.id,
            node.op,
            upstream,
            [
                TraceContribution(node.inputs[0], "upstream * (a >= b)", "(a >= b)", left_contrib),
                TraceContribution(node.inputs[1], "upstream * (b > a)", "(b > a)", right_contrib),
            ],
        )
    raise AutodiffError(f"Unsupported builtin '{name}' in autodiff")


def _backprop_rule(
    node, upstream, node_grads: Dict[int, np.ndarray], graph: Graph, rules: RuleTable
) -> TraceEntry:
    if node.rule_name is None:
        raise AutodiffError("Rule node missing rule name")
    rule = rules.get(node.rule_name)
    if len(node.inputs) != len(rule.params):
        raise AutodiffError(f"Rule call '{rule.name}' argument count mismatch")

    env: Dict[str, Value] = {}
    for param, node_id in zip(rule.params, node.inputs):
        env[param.name] = graph.nodes[node_id].value

    contributions: List[TraceContribution] = []
    for param, node_id in zip(rule.params, node.inputs):
        if param.nondiff:
            continue
        deriv_expr = rule.derivs.get(param.name)
        if deriv_expr is None:
            raise AutodiffError(f"Missing derivative for '{param.name}' in rule '{rule.name}'")
        local = _eval_expr_value(deriv_expr, env, rules).as_array()
        contrib = upstream * local
        contributions.append(
            TraceContribution(
                node_id, f"upstream * d/d{param.name}", f"d/d{param.name}", contrib
            )
        )
        _add_grad(node_grads, node_id, contrib)
    return TraceEntry(node.id, node.op, upstream, contributions)


def _add_grad(node_grads: Dict[int, np.ndarray], node_id: int, value: np.ndarray) -> None:
    if node_id in node_grads:
        node_grads[node_id] = node_grads[node_id] + value
    else:
        node_grads[node_id] = value


def _reduce_grad(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    g = np.asarray(grad)
    if shape == ():
        return np.array(g.sum())
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and g.shape[axis] != 1:
            g = g.sum(axis=axis, keepdims=True)
    if g.shape != shape:
        g = g.reshape(shape)
    return g


def _matmul_grads(
    upstream: np.ndarray, left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    up = np.asarray(upstream)
    if left.ndim == 1 and right.ndim == 1:
        return up * right, up * left
    if left.ndim == 1 and right.ndim == 2:
        left_contrib = up @ right.T
        right_contrib = np.outer(left, up)
        return left_contrib, right_contrib
    if left.ndim == 2 and right.ndim == 1:
        left_contrib = np.outer(up, right)
        right_contrib = left.T @ up
        return left_contrib, right_contrib
    if left.ndim == 2 and right.ndim == 2:
        return up @ right.T, left.T @ up
    raise AutodiffError("matmul gradients support only 1D/2D inputs")


def _param_name_to_id(graph: Graph) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for node in graph.nodes:
        if node.op == "param" and node.name:
            mapping[node.name] = node.id
    return mapping
