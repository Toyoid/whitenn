from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .. import ast


class RuleError(Exception):
    pass


@dataclass(frozen=True)
class Rule:
    name: str
    params: Sequence[ast.RuleParam]
    return_type: ast.TypeRef
    forward: ast.Expr
    derivs: Dict[str, ast.Expr]

    @property
    def param_names(self) -> List[str]:
        return [param.name for param in self.params]


class RuleTable:
    def __init__(self) -> None:
        self._rules: Dict[str, Rule] = {}

    def register(self, rule: Rule) -> None:
        if rule.name in self._rules:
            raise RuleError(f"Rule '{rule.name}' already exists")
        self._rules[rule.name] = rule

    def get(self, name: str) -> Rule:
        try:
            return self._rules[name]
        except KeyError as exc:
            raise RuleError(f"Unknown rule '{name}'") from exc

    def items(self) -> Iterable[tuple[str, Rule]]:
        return self._rules.items()

    @classmethod
    def from_program(cls, program: ast.Program) -> "RuleTable":
        table = cls()
        for item in program.items:
            if isinstance(item, ast.RuleDecl):
                table.register(compile_rule(item))
        return table


def compile_rule(rule_decl: ast.RuleDecl) -> Rule:
    forward_expr: ast.Expr | None = None
    derivs: Dict[str, ast.Expr] = {}
    param_names = {param.name for param in rule_decl.params}

    for clause in rule_decl.clauses:
        if clause.kind == "forward":
            if forward_expr is not None:
                raise RuleError(f"Multiple forward clauses in rule '{rule_decl.name}'")
            forward_expr = clause.expr
        elif clause.kind == "deriv":
            if clause.target is None:
                raise RuleError(f"Missing derivative target in rule '{rule_decl.name}'")
            if clause.target not in param_names:
                raise RuleError(
                    f"Derivative for unknown parameter '{clause.target}' in rule '{rule_decl.name}'"
                )
            if clause.target in derivs:
                raise RuleError(
                    f"Duplicate derivative for '{clause.target}' in rule '{rule_decl.name}'"
                )
            derivs[clause.target] = clause.expr
        else:
            raise RuleError(f"Unknown rule clause kind '{clause.kind}' in rule '{rule_decl.name}'")

    if forward_expr is None:
        raise RuleError(f"Missing forward clause in rule '{rule_decl.name}'")

    missing = [
        param.name
        for param in rule_decl.params
        if not param.nondiff and param.name not in derivs
    ]
    if missing:
        missing_list = ", ".join(missing)
        raise RuleError(f"Missing derivative clause(s): {missing_list} in rule '{rule_decl.name}'")

    return Rule(
        name=rule_decl.name,
        params=rule_decl.params,
        return_type=rule_decl.return_type,
        forward=forward_expr,
        derivs=derivs,
    )
