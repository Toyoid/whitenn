from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from . import ast


@dataclass(frozen=True)
class ValidationIssue:
    message: str
    context: Optional[str] = None


class ValidationError(Exception):
    def __init__(self, issues: Sequence[ValidationIssue]) -> None:
        self.issues = list(issues)
        super().__init__("\n".join(_format_issue(issue) for issue in self.issues))


def validate_program(program: ast.Program) -> None:
    validator = _Validator()
    validator.validate_program(program)
    if validator.issues:
        raise ValidationError(validator.issues)


def _format_issue(issue: ValidationIssue) -> str:
    if issue.context:
        return f"{issue.message} ({issue.context})"
    return issue.message


class _Validator:
    def __init__(self) -> None:
        self.issues: List[ValidationIssue] = []
        self._top_level_names: dict[str, str] = {}

    def validate_program(self, program: ast.Program) -> None:
        for item in program.items:
            if isinstance(item, ast.Decl):
                self._validate_decl(item)
            elif isinstance(item, ast.Stmt):
                self._validate_stmt(item)
            else:
                self._error("Unknown top-level item", type(item).__name__)

    def _validate_decl(self, decl: ast.Decl) -> None:
        if isinstance(decl, ast.ModelDecl):
            self._register_name(decl.name, "model")
            self._validate_model_decl(decl)
            return
        if isinstance(decl, ast.RuleDecl):
            self._register_name(decl.name, "rule")
            self._validate_rule_decl(decl)
            return
        if isinstance(decl, ast.FnDecl):
            self._register_name(decl.name, "fn")
            self._validate_fn_decl(decl)
            return
        self._error("Unknown declaration", type(decl).__name__)

    def _validate_stmt(self, stmt: ast.Stmt) -> None:
        if isinstance(stmt, ast.GraphStmt):
            self._validate_graph_stmt(stmt)
        elif isinstance(stmt, ast.GradStmt):
            self._validate_grad_stmt(stmt)
        elif isinstance(stmt, ast.StepStmt):
            self._validate_step_stmt(stmt)
        elif isinstance(stmt, ast.ExplainStmt):
            self._validate_explain_stmt(stmt)
        elif isinstance(stmt, ast.ForStmt):
            self._validate_for_stmt(stmt)
        elif isinstance(stmt, ast.AssignStmt):
            return
        elif isinstance(stmt, ast.ExprStmt):
            return
        else:
            self._error("Unknown statement", type(stmt).__name__)

    def _register_name(self, name: str, kind: str) -> None:
        existing = self._top_level_names.get(name)
        if existing:
            self._error(f"Duplicate top-level name '{name}'", f"{existing} vs {kind}")
        else:
            self._top_level_names[name] = kind

    def _validate_model_decl(self, decl: ast.ModelDecl) -> None:
        seen: set[str] = set()
        for param in decl.params:
            if param.name in seen:
                self._error(
                    f"Duplicate param name '{param.name}'",
                    f"model {decl.name}",
                )
            seen.add(param.name)
            if param.shape:
                self._validate_shape(param.shape, f"model {decl.name} param {param.name}")

    def _validate_rule_decl(self, decl: ast.RuleDecl) -> None:
        seen_params: set[str] = set()
        nondiff_params: set[str] = set()
        for param in decl.params:
            if param.name in seen_params:
                self._error(
                    f"Duplicate rule parameter '{param.name}'",
                    f"rule {decl.name}",
                )
            seen_params.add(param.name)
            if param.nondiff:
                nondiff_params.add(param.name)
            self._validate_type_ref(param.type_ref, f"rule {decl.name} param {param.name}")

        self._validate_type_ref(decl.return_type, f"rule {decl.name} return type")

        forward_count = 0
        deriv_targets: set[str] = set()
        for clause in decl.clauses:
            if clause.kind == "forward":
                forward_count += 1
                continue
            if clause.kind == "deriv":
                target = clause.target
                if target not in seen_params:
                    self._error(
                        f"Derivative for unknown parameter '{target}'",
                        f"rule {decl.name}",
                    )
                    continue
                if target in deriv_targets:
                    self._error(
                        f"Duplicate derivative clause for '{target}'",
                        f"rule {decl.name}",
                    )
                deriv_targets.add(target)
                if target in nondiff_params:
                    self._error(
                        f"Derivative provided for nondiff parameter '{target}'",
                        f"rule {decl.name}",
                    )
                continue
            self._error("Unknown rule clause kind", f"rule {decl.name}")

        if forward_count != 1:
            if forward_count == 0:
                self._error("Missing forward clause", f"rule {decl.name}")
            else:
                self._error("Multiple forward clauses", f"rule {decl.name}")

        missing = seen_params - nondiff_params - deriv_targets
        if missing:
            missing_list = ", ".join(sorted(missing))
            self._error(
                f"Missing derivative clause(s): {missing_list}",
                f"rule {decl.name}",
            )

    def _validate_fn_decl(self, decl: ast.FnDecl) -> None:
        seen_params: set[str] = set()
        for param in decl.params:
            if param.name in seen_params:
                self._error(
                    f"Duplicate fn parameter '{param.name}'",
                    f"fn {decl.name}",
                )
            seen_params.add(param.name)
            if param.type_ref:
                self._validate_type_ref(param.type_ref, f"fn {decl.name} param {param.name}")
            if param.shape:
                self._validate_shape(param.shape, f"fn {decl.name} param {param.name}")
                if param.type_ref and param.type_ref.shape:
                    self._error(
                        f"Conflicting shapes for parameter '{param.name}'",
                        f"fn {decl.name}",
                    )

        if decl.return_type:
            self._validate_type_ref(decl.return_type, f"fn {decl.name} return type")

    def _validate_graph_stmt(self, stmt: ast.GraphStmt) -> None:
        seen: set[str] = set()
        for item in stmt.items:
            if item.name in seen:
                self._error(
                    f"Duplicate graph assignment to '{item.name}'",
                    "graph block",
                )
            seen.add(item.name)

    def _validate_grad_stmt(self, stmt: ast.GradStmt) -> None:
        if not stmt.params:
            self._error("grad wrt list cannot be empty", f"grad {stmt.name}")
            return
        seen: set[str] = set()
        for param in stmt.params:
            if param in seen:
                self._error(
                    f"Duplicate parameter in grad list '{param}'",
                    f"grad {stmt.name}",
                )
            seen.add(param)

    def _validate_step_stmt(self, stmt: ast.StepStmt) -> None:
        if not isinstance(stmt.optimizer, ast.CallExpr):
            self._error("step expects an optimizer call", f"step using {stmt.grad_name}")

    def _validate_explain_stmt(self, stmt: ast.ExplainStmt) -> None:
        if stmt.level is not None and stmt.level < 0:
            self._error("explain level must be non-negative", f"explain {stmt.grad_name}")

    def _validate_for_stmt(self, stmt: ast.ForStmt) -> None:
        if not stmt.var_name:
            self._error("for loop variable cannot be empty", "for loop")
        for nested in stmt.body.stmts:
            self._validate_stmt(nested)

    def _validate_type_ref(self, type_ref: ast.TypeRef, context: str) -> None:
        if type_ref.shape:
            self._validate_shape(type_ref.shape, context)

    def _validate_shape(self, shape: Sequence[int], context: str) -> None:
        for dim in shape:
            if dim <= 0:
                self._error("Shape dimensions must be positive integers", context)

    def _error(self, message: str, context: Optional[str] = None) -> None:
        self.issues.append(ValidationIssue(message, context))
