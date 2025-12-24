from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from lark import Lark, Token, Transformer

from . import ast


_GRAMMAR_PATH = Path(__file__).with_name("grammar.lark")


def _build_parser() -> Lark:
    grammar_text = _GRAMMAR_PATH.read_text(encoding="utf-8")
    return Lark(grammar_text, parser="lalr", maybe_placeholders=False)


_PARSER = _build_parser()


def parse_program(source: str) -> ast.Program:
    tree = _PARSER.parse(source)
    return _ToAst().transform(tree)


class _ToAst(Transformer):
    def program(self, items: List[object]) -> ast.Program:
        return ast.Program(items)

    def item(self, items: List[object]) -> object:
        return items[0]

    def decl(self, items: List[object]) -> object:
        return items[0]

    def stmt(self, items: List[object]) -> object:
        return items[0]

    def model_decl(self, items: List[object]) -> ast.ModelDecl:
        name = items[0]
        params = items[1:]
        return ast.ModelDecl(name, params)

    def param_decl(self, items: List[object]) -> ast.ParamDecl:
        name = items[0]
        shape = None
        init = None
        for item in items[1:]:
            if isinstance(item, list):
                shape = item
            else:
                init = item
        return ast.ParamDecl(name, shape, init)

    def shape_spec(self, items: List[int]) -> List[int]:
        return [int(value) for value in items]

    def init_clause(self, items: List[ast.Expr]) -> ast.Expr:
        return items[0]

    def rule_decl(self, items: List[object]) -> ast.RuleDecl:
        name = items[0]
        idx = 1
        params: List[ast.RuleParam] = []
        if idx < len(items) and isinstance(items[idx], list):
            params = items[idx]
            idx += 1
        return_type = items[idx]
        clauses = items[idx + 1 :]
        return ast.RuleDecl(name, params, return_type, clauses)

    def rule_param_list(self, items: List[ast.RuleParam]) -> List[ast.RuleParam]:
        return items

    def rule_param(self, items: List[object]) -> ast.RuleParam:
        name = items[0]
        type_ref = items[1]
        nondiff = False
        if len(items) > 2:
            nondiff = True
        return ast.RuleParam(name, type_ref, nondiff)

    def param_attr(self, items: List[object]) -> bool:
        return True

    def rule_forward(self, items: List[ast.Expr]) -> ast.RuleClause:
        return ast.RuleClause("forward", None, items[0])

    def rule_deriv(self, items: List[object]) -> ast.RuleClause:
        target = items[0]
        expr = items[1]
        return ast.RuleClause("deriv", target, expr)

    def fn_decl(self, items: List[object]) -> ast.FnDecl:
        name = items[0]
        idx = 1
        params: List[ast.FnParam] = []
        if idx < len(items) and isinstance(items[idx], list):
            params = items[idx]
            idx += 1
        return_type = None
        if idx < len(items) and isinstance(items[idx], ast.TypeRef):
            return_type = items[idx]
            idx += 1
        body = items[idx]
        return ast.FnDecl(name, params, return_type, body)

    def fn_param_list(self, items: List[ast.FnParam]) -> List[ast.FnParam]:
        return items

    def fn_param(self, items: List[object]) -> ast.FnParam:
        name = items[0]
        type_ref = None
        shape = None
        for item in items[1:]:
            if isinstance(item, ast.TypeRef):
                type_ref = item
            elif isinstance(item, list):
                shape = item
        return ast.FnParam(name, type_ref, shape)

    def fn_return(self, items: List[ast.TypeRef]) -> ast.TypeRef:
        return items[0]

    def type_ref(self, items: List[object]) -> ast.TypeRef:
        name = items[0]
        shape = None
        if len(items) > 1:
            shape = items[1]
        return ast.TypeRef(name, shape)

    def block(self, items: List[ast.Stmt]) -> ast.Block:
        return ast.Block(items)

    def graph_stmt(self, items: List[ast.GraphItem]) -> ast.GraphStmt:
        return ast.GraphStmt(items)

    def graph_item(self, items: List[object]) -> ast.GraphItem:
        name = items[0]
        expr = items[1]
        return ast.GraphItem(name, expr)

    def assign_stmt(self, items: List[object]) -> ast.AssignStmt:
        return ast.AssignStmt(items[0], items[1])

    def grad_stmt(self, items: List[object]) -> ast.GradStmt:
        name = items[0]
        loss_name = items[1]
        params = items[2]
        return ast.GradStmt(name, loss_name, params)

    def ident_list(self, items: List[str]) -> List[str]:
        return items

    def explain_stmt(self, items: List[object]) -> ast.ExplainStmt:
        name = items[0]
        level = items[1] if len(items) > 1 else None
        return ast.ExplainStmt(name, level)

    def step_stmt(self, items: List[object]) -> ast.StepStmt:
        optimizer = items[0]
        grad_name = items[1]
        return ast.StepStmt(optimizer, grad_name)

    def for_stmt(self, items: List[object]) -> ast.ForStmt:
        var_name = items[0]
        start, end = items[1]
        body = items[2]
        return ast.ForStmt(var_name, start, end, body)

    def range_expr(self, items: List[ast.Expr]) -> Tuple[ast.Expr, ast.Expr]:
        return items[0], items[1]

    def expr_stmt(self, items: List[ast.Expr]) -> ast.ExprStmt:
        return ast.ExprStmt(items[0])

    def ternary(self, items: List[ast.Expr]) -> ast.Expr:
        if len(items) == 1:
            return items[0]
        return ast.TernaryOp(items[0], items[1], items[2])

    def compare(self, items: List[object]) -> ast.Expr:
        return _fold_bin(items)

    def sum(self, items: List[object]) -> ast.Expr:
        return _fold_bin(items)

    def term(self, items: List[object]) -> ast.Expr:
        return _fold_bin(items)

    def unary_neg(self, items: List[ast.Expr]) -> ast.UnaryOp:
        return ast.UnaryOp("-", items[0])

    def number(self, items: List[object]) -> ast.Number:
        value = float(items[0])
        return ast.Number(value)

    def name(self, items: List[str]) -> ast.Name:
        return ast.Name(items[0])

    def call_expr(self, items: List[object]) -> ast.CallExpr:
        func = items[0]
        args = items[1] if len(items) > 1 else []
        return ast.CallExpr(func, args)

    def arg_list(self, items: List[ast.CallArg]) -> List[ast.CallArg]:
        return items

    def arg_pos(self, items: List[ast.Expr]) -> ast.CallArg:
        return ast.CallArg(None, items[0])

    def arg_kw(self, items: List[object]) -> ast.CallArg:
        name = items[0]
        value = items[1]
        return ast.CallArg(name, value)

    def IDENT(self, token: Token) -> str:
        return str(token)

    def INT(self, token: Token) -> int:
        return int(token)

    def FLOAT(self, token: Token) -> float:
        return float(token)

    def COMP_OP(self, token: Token) -> str:
        return str(token)

    def ADD_OP(self, token: Token) -> str:
        return str(token)

    def MUL_OP(self, token: Token) -> str:
        return str(token)


def _fold_bin(items: List[object]) -> ast.Expr:
    if not items:
        raise ValueError("Empty binary expression")
    expr = items[0]
    idx = 1
    while idx < len(items):
        op = str(items[idx])
        right = items[idx + 1]
        expr = ast.BinaryOp(op, expr, right)
        idx += 2
    return expr
