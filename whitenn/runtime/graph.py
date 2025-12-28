from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import ast
from ..errors import RuntimeError, format_error
from .params import ParamStore
from .rules import RuleTable
from .values import Value


class GraphError(RuntimeError):
    pass


@dataclass
class Node:
    id: int
    op: str
    inputs: Sequence[int]
    value: Value
    name: Optional[str] = None
    rule_name: Optional[str] = None
    expr: Optional[ast.Expr] = None


@dataclass
class Graph:
    nodes: List[Node]
    name_to_id: Dict[str, int]

    def get(self, name: str) -> Node:
        try:
            node_id = self.name_to_id[name]
            return self.nodes[node_id]
        except KeyError as exc:
            raise GraphError(f"Unknown graph node '{name}'") from exc


class GraphExecutor:
    def __init__(
        self,
        rules: RuleTable,
        params: ParamStore,
        source: str | None = None,
        filename: str | None = None,
    ) -> None:
        self._rules = rules
        self._params = params
        self._source = source
        self._filename = filename
        self._value_env: Dict[str, object] = {}

    def execute(self, graph_stmt: ast.GraphStmt, env: Optional[Dict[str, object]] = None) -> Graph:
        builder = _GraphBuilder()
        local_env: Dict[str, int] = {}

        previous_env = self._value_env
        self._value_env = env or {}
        try:
            for item in graph_stmt.items:
                value, node_id = self._eval_expr(item.expr, local_env, builder)
                builder.name_to_id[item.name] = node_id
                if builder.nodes[node_id].name is None:
                    builder.nodes[node_id].name = item.name
                builder.nodes[node_id].expr = item.expr
                local_env[item.name] = node_id
        finally:
            self._value_env = previous_env

        return Graph(nodes=builder.nodes, name_to_id=builder.name_to_id)

    def extend(self, graph: Graph, name: str, expr: ast.Expr) -> Graph:
        builder = _GraphBuilder.from_graph(graph)
        env: Dict[str, int] = dict(builder.name_to_id)
        for pname, pid in builder.param_nodes.items():
            env[pname] = pid
        for iname, iid in builder.input_nodes.items():
            env[iname] = iid

        previous_env = self._value_env
        self._value_env = {}
        try:
            _, node_id = self._eval_expr(expr, env, builder)
            if name in builder.name_to_id and builder.name_to_id[name] != node_id:
                raise GraphError(f"Graph already has a different node named '{name}'")
            builder.name_to_id[name] = node_id
            if builder.nodes[node_id].name is None:
                builder.nodes[node_id].name = name
            builder.nodes[node_id].expr = expr
        finally:
            self._value_env = previous_env
        return Graph(nodes=builder.nodes, name_to_id=builder.name_to_id)

    def _eval_expr(
        self, expr: ast.Expr, env: Dict[str, int], builder: "_GraphBuilder"
    ) -> Tuple[Value, int]:
        try:
            return self._eval_expr_inner(expr, env, builder)
        except GraphError as exc:
            if isinstance(expr, ast.Expr):
                span = getattr(expr, "span", None)
            else:
                span = None
            message = format_error(
                str(exc), span=span, source=self._source, filename=self._filename
            )
            raise GraphError(message) from exc

    def _eval_expr_inner(
        self, expr: ast.Expr, env: Dict[str, int], builder: "_GraphBuilder"
    ) -> Tuple[Value, int]:
        if isinstance(expr, ast.Number):
            value = Value(float(expr.value))
            node_id = builder.add_node("const", [], value)
            return value, node_id
        if isinstance(expr, ast.StringLiteral):
            raise GraphError("String literal not allowed in graph expressions")
        if isinstance(expr, ast.IndexExpr):
            raise GraphError("Indexing is not allowed in graph expressions")
        if isinstance(expr, ast.ListLiteral):
            result = Value(_eval_list_literal(expr))
            node_id = builder.add_node("list", [], result)
            return result, node_id
        if isinstance(expr, ast.Name):
            return self._lookup_name(expr.value, env, builder)
        if isinstance(expr, ast.UnaryOp):
            value, child_id = self._eval_expr(expr.expr, env, builder)
            if expr.op != "-":
                raise GraphError(f"Unsupported unary operator '{expr.op}'")
            result = Value(-value.as_array())
            node_id = builder.add_node("unary:-", [child_id], result)
            return result, node_id
        if isinstance(expr, ast.BinaryOp):
            left, left_id = self._eval_expr(expr.left, env, builder)
            right, right_id = self._eval_expr(expr.right, env, builder)
            try:
                result = Value(_eval_binary(expr.op, left.as_array(), right.as_array()))
            except Exception as exc:
                raise GraphError(f"Binary op '{expr.op}' failed: {exc}") from exc
            node_id = builder.add_node(f"binop:{expr.op}", [left_id, right_id], result)
            return result, node_id
        if isinstance(expr, ast.TernaryOp):
            cond, cond_id = self._eval_expr(expr.cond, env, builder)
            then_value, then_id = self._eval_expr(expr.then_expr, env, builder)
            else_value, else_id = self._eval_expr(expr.else_expr, env, builder)
            result = Value(
                _eval_ternary(cond.as_array(), then_value.as_array(), else_value.as_array())
            )
            node_id = builder.add_node("ternary", [cond_id, then_id, else_id], result)
            return result, node_id
        if isinstance(expr, ast.CallExpr):
            return self._eval_call(expr, env, builder)
        raise GraphError(f"Unsupported expression type '{type(expr).__name__}'")

    def _eval_call(
        self, expr: ast.CallExpr, env: Dict[str, int], builder: "_GraphBuilder"
    ) -> Tuple[Value, int]:
        if expr.func in _BUILTINS:
            if any(arg.name is not None for arg in expr.args):
                raise GraphError(f"Built-in '{expr.func}' does not accept keyword arguments")
            arg_values: List[Value] = []
            arg_ids: List[int] = []
            for arg in expr.args:
                value, node_id = self._eval_expr(arg.value, env, builder)
                arg_values.append(value)
                arg_ids.append(node_id)
            result = Value(_BUILTINS[expr.func](*[v.as_array() for v in arg_values]))
            node_id = builder.add_node(f"call:{expr.func}", arg_ids, result)
            return result, node_id

        rule = self._rules.get(expr.func)
        bound = _bind_call_args_with_nodes(rule.params, expr.args, env, builder, self)
        arg_values = [value for value, _ in bound]
        arg_ids = [node_id for _, node_id in bound]
        value = _eval_expr_value(rule.forward, _bind_rule_env(rule.params, arg_values), self._rules)
        node_id = builder.add_node(f"rule:{rule.name}", arg_ids, value, rule_name=rule.name)
        return value, node_id

    def _lookup_name(
        self, name: str, env: Dict[str, int], builder: "_GraphBuilder"
    ) -> Tuple[Value, int]:
        if name in env:
            node_id = env[name]
            return builder.nodes[node_id].value, node_id
        if name in self._value_env:
            node_id = builder.add_input(name, _to_value(self._value_env[name], name))
            env[name] = node_id
            return builder.nodes[node_id].value, node_id
        try:
            return builder.add_param(name, self._params.get(name).value)
        except Exception as exc:
            raise GraphError(f"Unknown name '{name}'") from exc


def _to_value(value: object, name: str | None = None) -> Value:
    if isinstance(value, Value):
        return value
    if isinstance(value, (int, float, bool, np.ndarray, list, tuple)):
        array = np.array(value)
        if array.dtype == object:
            label = f" for '{name}'" if name else ""
            raise GraphError(f"Graph input{label} must be numeric")
        return Value(array)
    label = f" '{name}'" if name else ""
    raise GraphError(f"Unsupported graph input{label}")


def _eval_expr_value(
    expr: ast.Expr, env: Dict[str, Value], rules: RuleTable
) -> Value:
    if isinstance(expr, ast.Number):
        return Value(float(expr.value))
    if isinstance(expr, ast.StringLiteral):
        raise GraphError("String literal not allowed in rule evaluation")
    if isinstance(expr, ast.IndexExpr):
        raise GraphError("Indexing is not allowed in rule evaluation")
    if isinstance(expr, ast.ListLiteral):
        return Value(_eval_list_literal(expr))
    if isinstance(expr, ast.Name):
        try:
            return env[expr.value]
        except KeyError as exc:
            raise GraphError(f"Unknown name '{expr.value}' in rule evaluation") from exc
    if isinstance(expr, ast.UnaryOp):
        value = _eval_expr_value(expr.expr, env, rules)
        if expr.op != "-":
            raise GraphError(f"Unsupported unary operator '{expr.op}' in rule evaluation")
        return Value(-value.as_array())
    if isinstance(expr, ast.BinaryOp):
        left = _eval_expr_value(expr.left, env, rules)
        right = _eval_expr_value(expr.right, env, rules)
        return Value(_eval_binary(expr.op, left.as_array(), right.as_array()))
    if isinstance(expr, ast.TernaryOp):
        cond = _eval_expr_value(expr.cond, env, rules)
        then_value = _eval_expr_value(expr.then_expr, env, rules)
        else_value = _eval_expr_value(expr.else_expr, env, rules)
        return Value(_eval_ternary(cond.as_array(), then_value.as_array(), else_value.as_array()))
    if isinstance(expr, ast.CallExpr):
        if expr.func in _BUILTINS:
            if any(arg.name is not None for arg in expr.args):
                raise GraphError(f"Built-in '{expr.func}' does not accept keyword arguments")
            args = [_eval_expr_value(arg.value, env, rules).as_array() for arg in expr.args]
            return Value(_BUILTINS[expr.func](*args))
        rule = rules.get(expr.func)
        bound = _bind_call_args_values(rule.params, expr.args, env, rules)
        return _eval_expr_value(rule.forward, bound, rules)
    raise GraphError(f"Unsupported expression type '{type(expr).__name__}' in rule evaluation")


def _eval_list_literal(expr: ast.ListLiteral) -> np.ndarray:
    return np.array(_eval_list_items(expr.items))


def _eval_list_items(items: Sequence[ast.Expr]) -> List[object]:
    values: List[object] = []
    for item in items:
        if isinstance(item, ast.Number):
            values.append(float(item.value))
            continue
        if isinstance(item, ast.UnaryOp) and item.op == "-":
            if isinstance(item.expr, ast.Number):
                values.append(-float(item.expr.value))
                continue
            raise GraphError("List literals must contain only numeric literals")
        if isinstance(item, ast.ListLiteral):
            values.append(_eval_list_items(item.items))
            continue
        raise GraphError("List literals must contain only numeric literals")
    return values


def _bind_rule_env(params: Sequence[ast.RuleParam], values: Sequence[Value]) -> Dict[str, Value]:
    env: Dict[str, Value] = {}
    for param, value in zip(params, values):
        env[param.name] = value
    return env


def _eval_binary(op: str, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "/":
        return left / right
    if op == "@":
        return left @ right
    if op == "==":
        return left == right
    if op == ">":
        return left > right
    if op == "<":
        return left < right
    if op == ">=":
        return left >= right
    if op == "<=":
        return left <= right
    raise GraphError(f"Unsupported binary operator '{op}'")


def _eval_ternary(cond: np.ndarray, then_value: np.ndarray, else_value: np.ndarray) -> np.ndarray:
    if cond.shape == ():
        return then_value if bool(cond) else else_value
    return np.where(cond, then_value, else_value)


def _bind_call_args_with_nodes(
    params: Sequence[ast.RuleParam],
    args: Sequence[ast.CallArg],
    env: Dict[str, int],
    builder: "_GraphBuilder",
    executor: GraphExecutor,
) -> List[Tuple[Value, int]]:
    bindings: Dict[str, Tuple[Value, int]] = {}
    param_names = [param.name for param in params]
    positional = [arg for arg in args if arg.name is None]
    keywords = [arg for arg in args if arg.name is not None]

    if len(positional) > len(params):
        raise GraphError("Too many positional arguments")

    for param, arg in zip(params, positional):
        bindings[param.name] = executor._eval_expr(arg.value, env, builder)

    for arg in keywords:
        name = arg.name or ""
        if name not in param_names:
            raise GraphError(f"Unknown keyword argument '{name}'")
        if name in bindings:
            raise GraphError(f"Duplicate argument '{name}'")
        bindings[name] = executor._eval_expr(arg.value, env, builder)

    missing = [name for name in param_names if name not in bindings]
    if missing:
        missing_list = ", ".join(missing)
        raise GraphError(f"Missing arguments: {missing_list}")

    return [bindings[param.name] for param in params]


def _bind_call_args_values(
    params: Sequence[ast.RuleParam],
    args: Sequence[ast.CallArg],
    env: Dict[str, Value],
    rules: RuleTable,
) -> Dict[str, Value]:
    bindings: Dict[str, Value] = {}
    param_names = [param.name for param in params]
    positional = [arg for arg in args if arg.name is None]
    keywords = [arg for arg in args if arg.name is not None]

    if len(positional) > len(params):
        raise GraphError("Too many positional arguments")

    for param, arg in zip(params, positional):
        bindings[param.name] = _eval_expr_value(arg.value, env, rules)

    for arg in keywords:
        name = arg.name or ""
        if name not in param_names:
            raise GraphError(f"Unknown keyword argument '{name}'")
        if name in bindings:
            raise GraphError(f"Duplicate argument '{name}'")
        bindings[name] = _eval_expr_value(arg.value, env, rules)

    missing = [name for name in param_names if name not in bindings]
    if missing:
        missing_list = ", ".join(missing)
        raise GraphError(f"Missing arguments: {missing_list}")
    return bindings


def _builtin_max(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a, b)


def _builtin_sum(x: np.ndarray) -> np.ndarray:
    return np.sum(x)


def _builtin_mean(x: np.ndarray) -> np.ndarray:
    return np.mean(x)


def _builtin_transpose(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim < 2:
        return x
    return np.swapaxes(x, -1, -2)


def _builtin_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.size == 0:
        return x
    x_max = np.max(x, axis=-1, keepdims=True)
    exps = np.exp(x - x_max)
    denom = np.sum(exps, axis=-1, keepdims=True)
    return exps / denom


_BUILTINS = {
    "exp": np.exp,
    "max": _builtin_max,
    "sum": _builtin_sum,
    "mean": _builtin_mean,
    "transpose": _builtin_transpose,
    "softmax": _builtin_softmax,
    "log": np.log,
}


class _GraphBuilder:
    def __init__(self) -> None:
        self.nodes: List[Node] = []
        self.name_to_id: Dict[str, int] = {}
        self._param_nodes: Dict[str, int] = {}
        self._input_nodes: Dict[str, int] = {}

    @classmethod
    def from_graph(cls, graph: Graph) -> "_GraphBuilder":
        builder = cls()
        builder.nodes = list(graph.nodes)
        builder.name_to_id = dict(graph.name_to_id)
        for node in builder.nodes:
            if node.op == "param" and node.name:
                builder._param_nodes[node.name] = node.id
            if node.op == "input" and node.name:
                builder._input_nodes[node.name] = node.id
        return builder

    def add_node(
        self,
        op: str,
        inputs: Sequence[int],
        value: Value,
        name: Optional[str] = None,
        rule_name: Optional[str] = None,
    ) -> int:
        node_id = len(self.nodes)
        self.nodes.append(Node(node_id, op, list(inputs), value, name=name, rule_name=rule_name))
        return node_id

    def add_input(self, name: str, value: Value) -> int:
        if name in self._input_nodes:
            return self._input_nodes[name]
        node_id = self.add_node("input", [], value, name=name)
        self._input_nodes[name] = node_id
        self.name_to_id[name] = node_id
        return node_id

    def add_param(self, name: str, value: Value) -> Tuple[Value, int]:
        if name in self._param_nodes:
            node_id = self._param_nodes[name]
            return self.nodes[node_id].value, node_id
        node_id = self.add_node("param", [], value, name=name)
        self._param_nodes[name] = node_id
        return value, node_id

    @property
    def param_nodes(self) -> Dict[str, int]:
        return self._param_nodes

    @property
    def input_nodes(self) -> Dict[str, int]:
        return self._input_nodes
