from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import ast
from .errors import RuntimeError as WhiteNNRuntimeError, format_error
from .validate import validate_program
from .runtime.autodiff import Grad, derive
from .runtime.explain import explain
from .runtime.graph import Graph, GraphError, GraphExecutor
from .runtime.optim import SGD, Optimizer, OptimizerError
from .runtime.params import ParamError, ParamStore
from .runtime.values import Value
from .runtime.rules import RuleTable


class InterpreterError(WhiteNNRuntimeError):
    def __init__(self, message: str, formatted: bool = False) -> None:
        self.formatted = formatted
        super().__init__(message)


@dataclass
class ExecContext:
    env: Dict[str, object]
    last_graph: Optional[Graph] = None
    last_grad: Optional[Grad] = None
    last_explain: Optional[str] = None
    explain_outputs: List[str] = field(default_factory=list)
    loss_map: Dict[str, str] = field(default_factory=dict)
    return_value: Optional[object] = None


class Interpreter:
    def __init__(self, program: ast.Program) -> None:
        validate_program(program, source=program.source, filename=program.filename)
        self.program = program
        self.rules = RuleTable.from_program(program)
        self.params = ParamStore(seed=0)
        self._rng = np.random.default_rng(0)
        self._functions: Dict[str, ast.FnDecl] = {}
        self._load_program()

    def _load_program(self) -> None:
        for item in self.program.items:
            if isinstance(item, ast.ModelDecl):
                self._load_model(item)
            elif isinstance(item, ast.FnDecl):
                self._functions[item.name] = item

    def _load_model(self, model: ast.ModelDecl) -> None:
        for param in model.params:
            try:
                init_fn = self._resolve_init(param.init, node=param)
                self.params.add_param(param.name, param.shape, init=init_fn)
            except ParamError as exc:
                raise self._error(str(exc), node=param) from exc

    def run_function(self, name: str, args: Sequence[object] | Dict[str, object]) -> ExecContext:
        fn = self._functions.get(name)
        if fn is None:
            raise InterpreterError(f"Unknown function '{name}'")
        env = self._bind_args(fn, args)
        ctx = ExecContext(env=env)
        self._exec_block(fn.body, ctx)
        return ctx

    def run_program(self) -> ExecContext:
        ctx = ExecContext(env={})
        for item in self.program.items:
            if isinstance(item, ast.Stmt):
                self._exec_stmt(item, ctx)
        return ctx

    def _bind_args(
        self, fn: ast.FnDecl, args: Sequence[object] | Dict[str, object]
    ) -> Dict[str, object]:
        env: Dict[str, object] = {}
        if isinstance(args, dict):
            for param in fn.params:
                if param.name not in args:
                    raise InterpreterError(f"Missing argument '{param.name}'")
                env[param.name] = args[param.name]
            return env

        if len(args) != len(fn.params):
            raise InterpreterError(f"Expected {len(fn.params)} arguments, got {len(args)}")
        for param, value in zip(fn.params, args):
            env[param.name] = value
        return env

    def _exec_block(self, block: ast.Block, ctx: ExecContext) -> None:
        for stmt in block.stmts:
            self._exec_stmt(stmt, ctx)
            if ctx.return_value is not None:
                return

    def _exec_stmt(self, stmt: ast.Stmt, ctx: ExecContext) -> None:
        if isinstance(stmt, ast.GraphStmt):
            executor = GraphExecutor(
                self.rules, self.params, source=self.program.source, filename=self.program.filename
            )
            ctx.last_graph = executor.execute(stmt, env=ctx.env)
            return
        if isinstance(stmt, ast.LossStmt):
            if ctx.last_graph is None:
                raise self._error("loss requires a preceding graph", node=stmt)
            if not _loss_expr_is_graph_only(stmt.expr, ctx.last_graph.name_to_id):
                raise self._error(
                    "loss expression must reference graph-computed names only", node=stmt.expr
                )
            executor = GraphExecutor(
                self.rules, self.params, source=self.program.source, filename=self.program.filename
            )
            ctx.last_graph = executor.extend(ctx.last_graph, stmt.name, stmt.expr)
            ctx.loss_map[stmt.name] = stmt.name
            return
        if isinstance(stmt, ast.GradStmt):
            if ctx.last_graph is None:
                raise self._error("grad requires a preceding graph", node=stmt)
            loss_name = ctx.loss_map.get(stmt.loss_name, stmt.loss_name)
            ctx.last_grad = derive(
                ctx.last_graph,
                loss_name,
                stmt.params,
                self.rules,
                source=self.program.source,
                filename=self.program.filename,
            )
            ctx.env[stmt.name] = ctx.last_grad
            return
        if isinstance(stmt, ast.StepStmt):
            grad = ctx.env.get(stmt.grad_name)
            if not isinstance(grad, Grad):
                raise self._error(f"Unknown grad '{stmt.grad_name}'", node=stmt)
            optimizer = self._eval_optimizer(stmt.optimizer, ctx)
            try:
                optimizer.apply(grad, self.params)
            except OptimizerError as exc:
                raise self._error(str(exc), node=stmt) from exc
            return
        if isinstance(stmt, ast.ExplainStmt):
            grad = ctx.env.get(stmt.grad_name)
            if not isinstance(grad, Grad):
                raise self._error(f"Unknown grad '{stmt.grad_name}'", node=stmt)
            level = stmt.level if stmt.level is not None else 0
            if level == 1 and stmt.output_path is None:
                raise self._error("Explain level 1 requires an output path", node=stmt)
            try:
                ctx.last_explain = explain(grad, level=level)
            except Exception as exc:
                raise self._error(str(exc), node=stmt) from exc
            if level != 1:
                ctx.explain_outputs.append(ctx.last_explain)
                print(ctx.last_explain)
            if stmt.output_path:
                out_value = self._eval_host_expr(stmt.output_path, ctx)
                if not isinstance(out_value, str):
                    raise self._error("Explain output path must be a string", node=stmt.output_path)
                try:
                    self._render_explain_output(ctx.last_explain, out_value)
                except Exception as exc:
                    raise self._error(str(exc), node=stmt) from exc
            return
        if isinstance(stmt, ast.FetchStmt):
            if ctx.last_graph is None:
                raise self._error("fetch requires a preceding graph", node=stmt)
            if self._is_param_name(stmt.target_name):
                raise self._error(f"Cannot assign to param '{stmt.target_name}'", node=stmt)
            try:
                node = ctx.last_graph.get(stmt.source_name)
            except GraphError as exc:
                raise self._error(str(exc), node=stmt) from exc
            value = node.value.as_array()
            ctx.env[stmt.target_name] = float(value) if value.shape == () else value
            return
        if isinstance(stmt, ast.ReturnStmt):
            if stmt.value is None:
                ctx.return_value = None
            else:
                ctx.return_value = self._eval_host_expr(stmt.value, ctx)
            return
        if isinstance(stmt, ast.AssignStmt):
            if self._is_param_name(stmt.name):
                raise self._error(f"Cannot assign to param '{stmt.name}'", node=stmt)
            ctx.env[stmt.name] = self._eval_host_expr(stmt.expr, ctx)
            return
        if isinstance(stmt, ast.ExprStmt):
            self._eval_host_expr(stmt.expr, ctx)
            return
        if isinstance(stmt, ast.ForStmt):
            self._exec_for(stmt, ctx)
            return
        raise InterpreterError(f"Unsupported statement '{type(stmt).__name__}'")

    def _exec_for(self, stmt: ast.ForStmt, ctx: ExecContext) -> None:
        try:
            start = int(self._eval_host_expr(stmt.start, ctx))
        except InterpreterError as exc:
            if exc.formatted:
                raise
            raise self._error(str(exc), node=stmt.start) from exc
        except Exception as exc:
            raise self._error(f"for loop start must be an integer: {exc}", node=stmt.start) from exc
        try:
            end = int(self._eval_host_expr(stmt.end, ctx))
        except InterpreterError as exc:
            if exc.formatted:
                raise
            raise self._error(str(exc), node=stmt.end) from exc
        except Exception as exc:
            raise self._error(f"for loop end must be an integer: {exc}", node=stmt.end) from exc
        for i in range(start, end + 1):
            ctx.env[stmt.var_name] = i
            self._exec_block(stmt.body, ctx)

    def _eval_optimizer(self, call: ast.CallExpr, ctx: ExecContext) -> Optimizer:
        if call.func != "SGD":
            raise self._error(f"Unknown optimizer '{call.func}'", node=call)
        lr = None
        for arg in call.args:
            value = self._eval_host_expr(arg.value, ctx)
            if arg.name == "lr":
                try:
                    lr = float(value)
                except Exception as exc:
                    raise self._error(f"SGD lr must be numeric: {exc}", node=arg.value) from exc
            elif arg.name is None:
                try:
                    lr = float(value)
                except Exception as exc:
                    raise self._error(f"SGD lr must be numeric: {exc}", node=arg.value) from exc
        if lr is None:
            raise self._error("SGD requires lr", node=call)
        return SGD(lr=lr)

    def _resolve_init(self, init_expr: Optional[ast.Expr], node: Optional[ast.ParamDecl] = None):
        try:
            if init_expr is None:
                return self.params.resolve_init("zeros", [])
            if isinstance(init_expr, ast.Name):
                return self.params.resolve_init(init_expr.value, [])
            if isinstance(init_expr, ast.CallExpr):
                args = [float(self._eval_const_expr(arg.value)) for arg in init_expr.args]
                return self.params.resolve_init(init_expr.func, args)
        except ParamError as exc:
            raise self._error(str(exc), node=init_expr or node) from exc
        raise self._error("Unsupported init expression", node=init_expr or node)

    def _eval_const_expr(self, expr: ast.Expr) -> float:
        if isinstance(expr, ast.Number):
            return float(expr.value)
        if isinstance(expr, ast.UnaryOp) and expr.op == "-":
            return -self._eval_const_expr(expr.expr)
        raise self._error("Initializer args must be numeric literals", node=expr)

    def _eval_host_expr(self, expr: ast.Expr, ctx: ExecContext) -> object:
        if isinstance(expr, ast.Number):
            return float(expr.value)
        if isinstance(expr, ast.StringLiteral):
            return expr.value
        if isinstance(expr, ast.ListLiteral):
            return [self._eval_host_expr(item, ctx) for item in expr.items]
        if isinstance(expr, ast.IndexExpr):
            base = self._eval_host_expr(expr.base, ctx)
            index = self._eval_host_expr(expr.index, ctx)
            try:
                idx = _coerce_index(index)
            except InterpreterError as exc:
                if exc.formatted:
                    raise
                raise self._error(str(exc), node=expr.index) from exc
            if isinstance(base, Value):
                base = base.as_array()
            try:
                return base[idx]
            except Exception as exc:
                raise self._error(f"Indexing failed: {exc}", node=expr) from exc
        if isinstance(expr, ast.Name):
            if expr.value in ctx.env:
                return ctx.env[expr.value]
            try:
                param = self.params.get(expr.value)
                return param.value.as_array()
            except ParamError:
                pass
            raise self._error(f"Unknown name '{expr.value}'", node=expr)
        if isinstance(expr, ast.UnaryOp) and expr.op == "-":
            value = self._eval_host_expr(expr.expr, ctx)
            try:
                return -float(value)
            except Exception as exc:
                raise self._error(f"Unary '-' expects a numeric value: {exc}", node=expr) from exc
        if isinstance(expr, ast.BinaryOp):
            left = self._eval_host_expr(expr.left, ctx)
            right = self._eval_host_expr(expr.right, ctx)
            try:
                return _eval_host_binop(expr.op, left, right)
            except InterpreterError as exc:
                if exc.formatted:
                    raise
                raise self._error(str(exc), node=expr) from exc
        if isinstance(expr, ast.CallExpr):
            try:
                if expr.func == "print":
                    values = [self._eval_host_expr(arg.value, ctx) for arg in expr.args]
                    print(*[_format_host_value(value) for value in values])
                    return None
                if expr.func == "sum":
                    args = _eval_host_positional_args(expr, ctx, self)
                    if len(args) != 1:
                        raise InterpreterError("sum expects exactly one argument")
                    return float(np.sum(np.asarray(args[0])))
                if expr.func == "mean":
                    args = _eval_host_positional_args(expr, ctx, self)
                    if len(args) != 1:
                        raise InterpreterError("mean expects exactly one argument")
                    return float(np.mean(np.asarray(args[0])))
                if expr.func == "argsort":
                    args = _eval_host_positional_args(expr, ctx, self)
                    if len(args) != 1:
                        raise InterpreterError("argsort expects exactly one argument")
                    data = np.asarray(args[0])
                    return np.argsort(data).astype(int)
                if expr.func == "argmax":
                    args = _eval_host_positional_args(expr, ctx, self)
                    if len(args) != 1:
                        raise InterpreterError("argmax expects exactly one argument")
                    data = np.asarray(args[0])
                    if data.size == 0:
                        raise InterpreterError("argmax expects a non-empty array")
                    return np.argmax(data, axis=-1)
                if expr.func == "one_hot":
                    args = _eval_host_positional_args(expr, ctx, self)
                    if len(args) != 2:
                        raise InterpreterError("one_hot expects indices and num_classes")
                    indices = np.asarray(args[0], dtype=int)
                    num_classes = _coerce_int(args[1], "one_hot num_classes")
                    out = np.zeros((indices.size, num_classes), dtype=float)
                    out[np.arange(indices.size), indices.ravel()] = 1.0
                    return out
                if expr.func == "len":
                    args = _eval_host_positional_args(expr, ctx, self)
                    if len(args) != 1:
                        raise InterpreterError("len expects exactly one argument")
                    return _host_len(args[0])
                if expr.func == "randn":
                    args = _eval_host_positional_args(expr, ctx, self)
                    shape = _coerce_shape(args)
                    return self._rng.normal(size=shape)
                if expr.func == "zeros":
                    args = _eval_host_positional_args(expr, ctx, self)
                    shape = _coerce_shape(args)
                    return np.zeros(shape, dtype=float)
                if expr.func == "ones":
                    args = _eval_host_positional_args(expr, ctx, self)
                    shape = _coerce_shape(args)
                    return np.ones(shape, dtype=float)
                if expr.func == "linspace":
                    args = _eval_host_positional_args(expr, ctx, self)
                    if len(args) != 3:
                        raise InterpreterError("linspace expects start, stop, num")
                    start = float(args[0])
                    stop = float(args[1])
                    num = _coerce_int(args[2], "linspace num")
                    return np.linspace(start, stop, num=num)
                if expr.func in self._functions:
                    args = [self._eval_host_expr(arg.value, ctx) for arg in expr.args]
                    fn_ctx = self.run_function(expr.func, args)
                    if fn_ctx.explain_outputs:
                        ctx.explain_outputs.extend(fn_ctx.explain_outputs)
                        ctx.last_explain = fn_ctx.last_explain
                    return fn_ctx.return_value
                raise InterpreterError(f"Unknown function '{expr.func}'")
            except InterpreterError as exc:
                if exc.formatted:
                    raise
                raise self._error(str(exc), node=expr) from exc
            except Exception as exc:
                raise self._error(str(exc), node=expr) from exc
        raise self._error(f"Unsupported expression '{type(expr).__name__}'", node=expr)

    def _is_param_name(self, name: str) -> bool:
        try:
            self.params.get(name)
            return True
        except ParamError:
            return False

    def _render_explain_output(self, explain_text: str, output_path: str) -> None:
        from .runtime.explain import render_dot

        idx = explain_text.find("digraph G {")
        if idx == -1:
            raise InterpreterError("No DOT graph found in explain output")
        dot = explain_text[idx:]
        fmt = "svg" if output_path.endswith(".svg") else "png" if output_path.endswith(".png") else ""
        if not fmt:
            raise InterpreterError("Explain output path must end with .svg or .png")
        render_dot(dot, output_path, fmt=fmt)

    def _error(
        self, message: str, *, node: Optional[ast.Stmt | ast.Expr | ast.Decl] = None
    ) -> InterpreterError:
        span = getattr(node, "span", None) if node is not None else None
        formatted = format_error(
            message,
            span=span,
            source=self.program.source,
            filename=self.program.filename,
        )
        return InterpreterError(formatted, formatted=True)


def _eval_host_binop(op: str, left: object, right: object) -> object:
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "/":
        return left / right
    if op == "==":
        return left == right
    raise InterpreterError(f"Unsupported host operator '{op}'")


def _format_host_value(value: object) -> object:
    if isinstance(value, Value):
        value = value.as_array()
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return float(value)
        return np.array2string(value, precision=4, separator=",")
    return value


def _eval_host_positional_args(
    call: ast.CallExpr, ctx: ExecContext, interpreter: "Interpreter"
) -> List[object]:
    if any(arg.name is not None for arg in call.args):
        raise InterpreterError(f"{call.func} does not accept keyword arguments")
    return [interpreter._eval_host_expr(arg.value, ctx) for arg in call.args]


def _coerce_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
    if isinstance(value, np.ndarray) and value.shape == ():
        return _coerce_int(value.item(), label)
    raise InterpreterError(f"{label} must be an integer")


def _coerce_index(value: object) -> int:
    return _coerce_int(value, "Index")


def _coerce_shape(values: Sequence[object]) -> tuple[int, ...]:
    if not values:
        raise InterpreterError("Shape must have at least one dimension")
    dims = tuple(_coerce_int(value, "Shape dimension") for value in values)
    return dims


def _host_len(value: object) -> int:
    if isinstance(value, Value):
        value = value.as_array()
    if isinstance(value, np.ndarray):
        return int(value.shape[0]) if value.shape else 0
    return len(value)


def _loss_expr_is_graph_only(expr: ast.Expr, graph_names: Dict[str, int]) -> bool:
    if isinstance(expr, ast.Number):
        return True
    if isinstance(expr, ast.StringLiteral):
        return False
    if isinstance(expr, ast.IndexExpr):
        return False
    if isinstance(expr, ast.ListLiteral):
        return all(_loss_expr_is_graph_only(item, graph_names) for item in expr.items)
    if isinstance(expr, ast.Name):
        return expr.value in graph_names
    if isinstance(expr, ast.UnaryOp):
        return _loss_expr_is_graph_only(expr.expr, graph_names)
    if isinstance(expr, ast.BinaryOp):
        return _loss_expr_is_graph_only(expr.left, graph_names) and _loss_expr_is_graph_only(
            expr.right, graph_names
        )
    if isinstance(expr, ast.TernaryOp):
        return (
            _loss_expr_is_graph_only(expr.cond, graph_names)
            and _loss_expr_is_graph_only(expr.then_expr, graph_names)
            and _loss_expr_is_graph_only(expr.else_expr, graph_names)
        )
    if isinstance(expr, ast.CallExpr):
        return all(_loss_expr_is_graph_only(arg.value, graph_names) for arg in expr.args)
    return False
