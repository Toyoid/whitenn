from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


# ---- Program and declarations ----


class Decl:
    pass


@dataclass(frozen=True)
class ModelDecl(Decl):
    name: str
    params: Sequence[ParamDecl]


@dataclass(frozen=True)
class ParamDecl:
    name: str
    shape: Optional[Sequence[int]]
    init: Optional["Expr"]


@dataclass(frozen=True)
class RuleDecl(Decl):
    name: str
    params: Sequence[RuleParam]
    return_type: "TypeRef"
    clauses: Sequence["RuleClause"]


@dataclass(frozen=True)
class RuleParam:
    name: str
    type_ref: "TypeRef"
    nondiff: bool


@dataclass(frozen=True)
class RuleClause:
    kind: str  # "forward" or "deriv"
    target: Optional[str]
    expr: "Expr"


@dataclass(frozen=True)
class FnDecl(Decl):
    name: str
    params: Sequence["FnParam"]
    return_type: Optional["TypeRef"]
    body: "Block"


@dataclass(frozen=True)
class FnParam:
    name: str
    type_ref: Optional["TypeRef"]
    shape: Optional[Sequence[int]]


@dataclass(frozen=True)
class TypeRef:
    name: str
    shape: Optional[Sequence[int]]


# ---- Statements ----


class Stmt:
    pass


@dataclass(frozen=True)
class Block:
    stmts: Sequence[Stmt]


@dataclass(frozen=True)
class GraphStmt(Stmt):
    items: Sequence["GraphItem"]


@dataclass(frozen=True)
class GraphItem:
    name: str
    expr: "Expr"


@dataclass(frozen=True)
class AssignStmt(Stmt):
    name: str
    expr: "Expr"


@dataclass(frozen=True)
class ExprStmt(Stmt):
    expr: "Expr"


@dataclass(frozen=True)
class GradStmt(Stmt):
    name: str
    loss_name: str
    params: Sequence[str]


@dataclass(frozen=True)
class StepStmt(Stmt):
    optimizer: "CallExpr"
    grad_name: str


@dataclass(frozen=True)
class ExplainStmt(Stmt):
    grad_name: str
    level: Optional[int]


@dataclass(frozen=True)
class ForStmt(Stmt):
    var_name: str
    start: "Expr"
    end: "Expr"
    body: Block


# ---- Expressions ----


class Expr:
    pass


@dataclass(frozen=True)
class Number(Expr):
    value: float


@dataclass(frozen=True)
class Name(Expr):
    value: str


@dataclass(frozen=True)
class CallExpr(Expr):
    func: str
    args: Sequence["CallArg"]


@dataclass(frozen=True)
class CallArg:
    name: Optional[str]
    value: Expr


@dataclass(frozen=True)
class UnaryOp(Expr):
    op: str
    expr: Expr


@dataclass(frozen=True)
class BinaryOp(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass(frozen=True)
class TernaryOp(Expr):
    cond: Expr
    then_expr: Expr
    else_expr: Expr


TopLevel = Decl | Stmt


@dataclass(frozen=True)
class Program:
    items: Sequence[TopLevel]
