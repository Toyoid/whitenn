from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


# ---- Program and declarations ----


class Decl:
    pass


@dataclass(frozen=True)
class Span:
    line: int
    column: int
    end_line: int
    end_column: int


@dataclass(frozen=True)
class ModelDecl(Decl):
    name: str
    params: Sequence[ParamDecl]
    span: Optional[Span] = None


@dataclass(frozen=True)
class ParamDecl:
    name: str
    shape: Optional[Sequence[int]]
    init: Optional["Expr"]
    span: Optional[Span] = None


@dataclass(frozen=True)
class RuleDecl(Decl):
    name: str
    params: Sequence[RuleParam]
    return_type: "TypeRef"
    clauses: Sequence["RuleClause"]
    span: Optional[Span] = None


@dataclass(frozen=True)
class RuleParam:
    name: str
    type_ref: "TypeRef"
    nondiff: bool
    span: Optional[Span] = None


@dataclass(frozen=True)
class RuleClause:
    kind: str  # "forward" or "deriv"
    target: Optional[str]
    expr: "Expr"
    span: Optional[Span] = None


@dataclass(frozen=True)
class FnDecl(Decl):
    name: str
    params: Sequence["FnParam"]
    return_type: Optional["TypeRef"]
    body: "Block"
    span: Optional[Span] = None


@dataclass(frozen=True)
class FnParam:
    name: str
    type_ref: Optional["TypeRef"]
    shape: Optional[Sequence[int]]
    span: Optional[Span] = None


@dataclass(frozen=True)
class TypeRef:
    name: str
    shape: Optional[Sequence[int]]
    span: Optional[Span] = None


# ---- Statements ----


class Stmt:
    pass


@dataclass(frozen=True)
class Block:
    stmts: Sequence[Stmt]
    span: Optional[Span] = None


@dataclass(frozen=True)
class GraphStmt(Stmt):
    items: Sequence["GraphItem"]
    span: Optional[Span] = None


@dataclass(frozen=True)
class GraphItem:
    name: str
    expr: "Expr"
    span: Optional[Span] = None


@dataclass(frozen=True)
class AssignStmt(Stmt):
    name: str
    expr: "Expr"
    span: Optional[Span] = None


@dataclass(frozen=True)
class ExprStmt(Stmt):
    expr: "Expr"
    span: Optional[Span] = None


@dataclass(frozen=True)
class GradStmt(Stmt):
    name: str
    loss_name: str
    params: Sequence[str]
    span: Optional[Span] = None


@dataclass(frozen=True)
class FetchStmt(Stmt):
    target_name: str
    source_name: str
    span: Optional[Span] = None


@dataclass(frozen=True)
class ReturnStmt(Stmt):
    value: Optional["Expr"]
    span: Optional[Span] = None


@dataclass(frozen=True)
class LossStmt(Stmt):
    name: str
    expr: "Expr"
    span: Optional[Span] = None


@dataclass(frozen=True)
class StepStmt(Stmt):
    optimizer: "CallExpr"
    grad_name: str
    span: Optional[Span] = None


@dataclass(frozen=True)
class ExplainStmt(Stmt):
    grad_name: str
    level: Optional[int]
    span: Optional[Span] = None
    output_path: Optional["Expr"] = None


@dataclass(frozen=True)
class ForStmt(Stmt):
    var_name: str
    start: "Expr"
    end: "Expr"
    body: Block
    span: Optional[Span] = None


# ---- Expressions ----


class Expr:
    pass


@dataclass(frozen=True)
class Number(Expr):
    value: float
    span: Optional[Span] = None


@dataclass(frozen=True)
class StringLiteral(Expr):
    value: str
    span: Optional[Span] = None


@dataclass(frozen=True)
class ListLiteral(Expr):
    items: Sequence[Expr]
    span: Optional[Span] = None


@dataclass(frozen=True)
class IndexExpr(Expr):
    base: Expr
    index: Expr
    span: Optional[Span] = None


@dataclass(frozen=True)
class Name(Expr):
    value: str
    span: Optional[Span] = None


@dataclass(frozen=True)
class CallExpr(Expr):
    func: str
    args: Sequence["CallArg"]
    span: Optional[Span] = None


@dataclass(frozen=True)
class CallArg:
    name: Optional[str]
    value: Expr
    span: Optional[Span] = None


@dataclass(frozen=True)
class UnaryOp(Expr):
    op: str
    expr: Expr
    span: Optional[Span] = None


@dataclass(frozen=True)
class BinaryOp(Expr):
    op: str
    left: Expr
    right: Expr
    span: Optional[Span] = None


@dataclass(frozen=True)
class TernaryOp(Expr):
    cond: Expr
    then_expr: Expr
    else_expr: Expr
    span: Optional[Span] = None


TopLevel = Decl | Stmt


@dataclass(frozen=True)
class Program:
    items: Sequence[TopLevel]
    source: Optional[str] = None
    filename: Optional[str] = None
    span: Optional[Span] = None
