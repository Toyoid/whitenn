from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from . import ast
from .errors import SemanticError, format_error


@dataclass(frozen=True)
class ValidationIssue:
    message: str
    context: Optional[str] = None
    span: Optional[ast.Span] = None


class ValidationError(SemanticError):
    def __init__(
        self,
        issues: Sequence[ValidationIssue],
        source: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> None:
        self.issues = list(issues)
        self.source = source
        self.filename = filename
        super().__init__(
            "\n".join(
                _format_issue(issue, source=source, filename=filename)
                for issue in self.issues
            )
        )


def validate_program(
    program: ast.Program, source: Optional[str] = None, filename: Optional[str] = None
) -> None:
    if source is None:
        source = getattr(program, "source", None)
    if filename is None:
        filename = getattr(program, "filename", None)
    validator = _Validator()
    validator.validate_program(program)
    if validator.issues:
        raise ValidationError(validator.issues, source=source, filename=filename)


def _format_issue(
    issue: ValidationIssue, source: Optional[str], filename: Optional[str]
) -> str:
    return format_error(
        issue.message,
        span=issue.span,
        source=source,
        filename=filename,
        context=issue.context,
    )


class _Validator:
    def __init__(self) -> None:
        self.issues: List[ValidationIssue] = []
        self._top_level_names: dict[str, str] = {}
        self._ctx = _BlockContext()
        self._param_names: set[str] = set()

    def validate_program(self, program: ast.Program) -> None:
        for item in program.items:
            if isinstance(item, ast.Decl):
                self._validate_decl(item)
            elif isinstance(item, ast.Stmt):
                self._validate_stmt(item, self._ctx)
            else:
                self._error("Unknown top-level item", type(item).__name__, node=item)

    def _validate_decl(self, decl: ast.Decl) -> None:
        if isinstance(decl, ast.ModelDecl):
            self._register_name(decl.name, "model", node=decl)
            self._validate_model_decl(decl)
            return
        if isinstance(decl, ast.RuleDecl):
            self._register_name(decl.name, "rule", node=decl)
            self._validate_rule_decl(decl)
            return
        if isinstance(decl, ast.FnDecl):
            self._register_name(decl.name, "fn", node=decl)
            self._validate_fn_decl(decl)
            return
        self._error("Unknown declaration", type(decl).__name__, node=decl)

    def _validate_stmt(self, stmt: ast.Stmt, ctx: "_BlockContext") -> None:
        if isinstance(stmt, ast.GraphStmt):
            self._validate_graph_stmt(stmt)
            ctx.last_graph_names = {item.name for item in stmt.items}
        elif isinstance(stmt, ast.LossStmt):
            self._validate_loss_stmt(stmt, ctx)
        elif isinstance(stmt, ast.GradStmt):
            self._validate_grad_stmt(stmt)
        elif isinstance(stmt, ast.StepStmt):
            self._validate_step_stmt(stmt)
        elif isinstance(stmt, ast.FetchStmt):
            self._validate_fetch_stmt(stmt, ctx)
        elif isinstance(stmt, ast.ReturnStmt):
            return
        elif isinstance(stmt, ast.ExplainStmt):
            self._validate_explain_stmt(stmt)
        elif isinstance(stmt, ast.ForStmt):
            self._validate_for_stmt(stmt, ctx)
        elif isinstance(stmt, ast.AssignStmt):
            if stmt.name in self._param_names:
                self._error(
                    f"Cannot assign to param '{stmt.name}'",
                    "assignment",
                    node=stmt,
                )
            return
        elif isinstance(stmt, ast.ExprStmt):
            return
        else:
            self._error("Unknown statement", type(stmt).__name__, node=stmt)

    def _register_name(self, name: str, kind: str, node: Optional[object] = None) -> None:
        existing = self._top_level_names.get(name)
        if existing:
            self._error(
                f"Duplicate top-level name '{name}'",
                f"{existing} vs {kind}",
                node=node,
            )
        else:
            self._top_level_names[name] = kind

    def _validate_model_decl(self, decl: ast.ModelDecl) -> None:
        seen: set[str] = set()
        for param in decl.params:
            if param.name in seen:
                self._error(
                    f"Duplicate param name '{param.name}'",
                    f"model {decl.name}",
                    node=param,
                )
            seen.add(param.name)
            self._param_names.add(param.name)
            if param.shape:
                self._validate_shape(
                    param.shape, f"model {decl.name} param {param.name}", node=param
                )

    def _validate_rule_decl(self, decl: ast.RuleDecl) -> None:
        seen_params: set[str] = set()
        nondiff_params: set[str] = set()
        for param in decl.params:
            if param.name in seen_params:
                self._error(
                    f"Duplicate rule parameter '{param.name}'",
                    f"rule {decl.name}",
                    node=param,
                )
            seen_params.add(param.name)
            if param.nondiff:
                nondiff_params.add(param.name)
            self._validate_type_ref(
                param.type_ref, f"rule {decl.name} param {param.name}", node=param
            )

        self._validate_type_ref(decl.return_type, f"rule {decl.name} return type", node=decl)

        forward_count = 0
        deriv_targets: set[str] = set()
        for clause in decl.clauses:
            if clause.kind == "forward":
                if _contains_index_expr(clause.expr):
                    self._error(
                        "Indexing is not allowed in rule expressions",
                        f"rule {decl.name}",
                        node=clause,
                    )
                forward_count += 1
                continue
            if clause.kind == "deriv":
                if _contains_index_expr(clause.expr):
                    self._error(
                        "Indexing is not allowed in rule expressions",
                        f"rule {decl.name}",
                        node=clause,
                    )
                target = clause.target
                if target not in seen_params:
                    self._error(
                        f"Derivative for unknown parameter '{target}'",
                        f"rule {decl.name}",
                        node=clause,
                    )
                    continue
                if target in deriv_targets:
                    self._error(
                        f"Duplicate derivative clause for '{target}'",
                        f"rule {decl.name}",
                        node=clause,
                    )
                deriv_targets.add(target)
                if target in nondiff_params:
                    self._error(
                        f"Derivative provided for nondiff parameter '{target}'",
                        f"rule {decl.name}",
                        node=clause,
                    )
                continue
            self._error("Unknown rule clause kind", f"rule {decl.name}", node=clause)

        if forward_count != 1:
            if forward_count == 0:
                self._error("Missing forward clause", f"rule {decl.name}", node=decl)
            else:
                self._error("Multiple forward clauses", f"rule {decl.name}", node=decl)

        missing = seen_params - nondiff_params - deriv_targets
        if missing:
            missing_list = ", ".join(sorted(missing))
            self._error(
                f"Missing derivative clause(s): {missing_list}",
                f"rule {decl.name}",
                node=decl,
            )

    def _validate_fn_decl(self, decl: ast.FnDecl) -> None:
        seen_params: set[str] = set()
        for param in decl.params:
            if param.name in seen_params:
                self._error(
                    f"Duplicate fn parameter '{param.name}'",
                    f"fn {decl.name}",
                    node=param,
                )
            seen_params.add(param.name)
            if param.type_ref:
                self._validate_type_ref(
                    param.type_ref, f"fn {decl.name} param {param.name}", node=param
                )
            if param.shape:
                self._validate_shape(
                    param.shape, f"fn {decl.name} param {param.name}", node=param
                )
                if param.type_ref and param.type_ref.shape:
                    self._error(
                        f"Conflicting shapes for parameter '{param.name}'",
                        f"fn {decl.name}",
                        node=param,
                    )

        if decl.return_type:
            self._validate_type_ref(decl.return_type, f"fn {decl.name} return type", node=decl)
        ctx = _BlockContext()
        self._validate_block(decl.body, ctx)

    def _validate_graph_stmt(self, stmt: ast.GraphStmt) -> None:
        seen: set[str] = set()
        for item in stmt.items:
            if _contains_index_expr(item.expr):
                self._error(
                    "Indexing is not allowed in graph expressions",
                    f"graph {item.name}",
                    node=item,
                )
            if item.name in seen:
                self._error(
                    f"Duplicate graph assignment to '{item.name}'",
                    "graph block",
                    node=item,
                )
            seen.add(item.name)

    def _validate_grad_stmt(self, stmt: ast.GradStmt) -> None:
        if not stmt.params:
            self._error("grad wrt list cannot be empty", f"grad {stmt.name}", node=stmt)
            return
        seen: set[str] = set()
        for param in stmt.params:
            if param in seen:
                self._error(
                    f"Duplicate parameter in grad list '{param}'",
                    f"grad {stmt.name}",
                    node=stmt,
                )
            seen.add(param)


    def _validate_step_stmt(self, stmt: ast.StepStmt) -> None:
        if not isinstance(stmt.optimizer, ast.CallExpr):
            self._error("step expects an optimizer call", f"step using {stmt.grad_name}", node=stmt)

    def _validate_explain_stmt(self, stmt: ast.ExplainStmt) -> None:
        if stmt.level is not None and stmt.level < 0:
            self._error(
                "explain level must be non-negative", f"explain {stmt.grad_name}", node=stmt
            )
        if stmt.output_path is not None and not isinstance(stmt.output_path, ast.Expr):
            self._error("explain output path must be an expression", "explain", node=stmt)

    def _validate_for_stmt(self, stmt: ast.ForStmt, ctx: "_BlockContext") -> None:
        if not stmt.var_name:
            self._error("for loop variable cannot be empty", "for loop", node=stmt)
        self._validate_block(stmt.body, ctx)

    def _validate_block(self, block: ast.Block, ctx: "_BlockContext") -> None:
        for stmt in block.stmts:
            self._validate_stmt(stmt, ctx)

    def _validate_loss_stmt(self, stmt: ast.LossStmt, ctx: "_BlockContext") -> None:
        if ctx.last_graph_names is None:
            self._error("loss requires a preceding graph", f"loss {stmt.name}", node=stmt)
            return
        if _contains_index_expr(stmt.expr):
            self._error(
                "Indexing is not allowed in loss expressions",
                f"loss {stmt.name}",
                node=stmt,
            )
            return
        names = _expr_names(stmt.expr)
        unknown = sorted(name for name in names if name not in ctx.last_graph_names)
        if unknown:
            missing = ", ".join(unknown)
            self._error(
                f"loss references non-graph name(s): {missing}",
                f"loss {stmt.name}",
                node=stmt,
            )
        ctx.last_graph_names.add(stmt.name)

    def _validate_fetch_stmt(self, stmt: ast.FetchStmt, ctx: "_BlockContext") -> None:
        if ctx.last_graph_names is None:
            self._error("fetch requires a preceding graph", f"fetch {stmt.target_name}", node=stmt)
            return
        if stmt.source_name not in ctx.last_graph_names:
            self._error(
                f"fetch references unknown graph name '{stmt.source_name}'",
                f"fetch {stmt.target_name}",
                node=stmt,
            )

    def _validate_type_ref(
        self, type_ref: ast.TypeRef, context: str, node: Optional[object] = None
    ) -> None:
        if type_ref.shape:
            self._validate_shape(type_ref.shape, context, node=node or type_ref)

    def _validate_shape(
        self, shape: Sequence[int], context: str, node: Optional[object] = None
    ) -> None:
        for dim in shape:
            if dim <= 0:
                self._error("Shape dimensions must be positive integers", context, node=node)

    def _error(
        self,
        message: str,
        context: Optional[str] = None,
        node: Optional[object] = None,
    ) -> None:
        span = getattr(node, "span", None)
        self.issues.append(ValidationIssue(message, context, span))


@dataclass
class _BlockContext:
    last_graph_names: Optional[set[str]] = None


def _expr_names(expr: ast.Expr) -> set[str]:
    names: set[str] = set()

    def _visit(node: ast.Expr) -> None:
        if isinstance(node, ast.Name):
            names.add(node.value)
        elif isinstance(node, ast.Number):
            return
        elif isinstance(node, ast.StringLiteral):
            return
        elif isinstance(node, ast.ListLiteral):
            for item in node.items:
                _visit(item)
        elif isinstance(node, ast.IndexExpr):
            _visit(node.base)
            _visit(node.index)
        elif isinstance(node, ast.UnaryOp):
            _visit(node.expr)
        elif isinstance(node, ast.BinaryOp):
            _visit(node.left)
            _visit(node.right)
        elif isinstance(node, ast.TernaryOp):
            _visit(node.cond)
            _visit(node.then_expr)
            _visit(node.else_expr)
        elif isinstance(node, ast.CallExpr):
            for arg in node.args:
                _visit(arg.value)

    _visit(expr)
    return names


def _contains_index_expr(expr: ast.Expr) -> bool:
    found = False

    def _visit(node: ast.Expr) -> None:
        nonlocal found
        if found:
            return
        if isinstance(node, ast.IndexExpr):
            found = True
            return
        if isinstance(node, ast.Name):
            return
        if isinstance(node, ast.Number):
            return
        if isinstance(node, ast.StringLiteral):
            return
        if isinstance(node, ast.ListLiteral):
            for item in node.items:
                _visit(item)
            return
        if isinstance(node, ast.UnaryOp):
            _visit(node.expr)
            return
        if isinstance(node, ast.BinaryOp):
            _visit(node.left)
            _visit(node.right)
            return
        if isinstance(node, ast.TernaryOp):
            _visit(node.cond)
            _visit(node.then_expr)
            _visit(node.else_expr)
            return
        if isinstance(node, ast.CallExpr):
            for arg in node.args:
                _visit(arg.value)
            return

    _visit(expr)
    return found
