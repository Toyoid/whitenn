from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from lark import Lark, Token, Transformer
from lark.exceptions import UnexpectedInput
from lark.visitors import v_args

from . import ast
from .errors import ParseError, format_error


_GRAMMAR_PATH = Path(__file__).with_name("grammar.lark")


def _build_parser() -> Lark:
    grammar_text = _GRAMMAR_PATH.read_text(encoding="utf-8")
    return Lark(
        grammar_text,
        parser="lalr",
        maybe_placeholders=False,
        propagate_positions=True,
    )


_PARSER = _build_parser()


def parse_program(source: str, filename: str | None = None) -> ast.Program:
    try:
        tree = _PARSER.parse(source)
    except UnexpectedInput as exc:
        span = ast.Span(exc.line, exc.column, exc.line, exc.column)
        message = format_error(
            "Syntax error", span=span, source=source, filename=filename
        )
        raise ParseError(message) from exc
    program = _ToAst().transform(tree)
    return ast.Program(
        items=program.items,
        source=source,
        filename=filename,
        span=program.span,
    )


def _span(meta) -> ast.Span:
    return ast.Span(meta.line, meta.column, meta.end_line, meta.end_column)


@v_args(meta=True)
class _ToAst(Transformer):
    def program(self, meta, items: List[object]) -> ast.Program:
        return ast.Program(items, span=_span(meta))

    def item(self, meta, items: List[object]) -> object:
        return items[0]

    def decl(self, meta, items: List[object]) -> object:
        return items[0]

    def stmt(self, meta, items: List[object]) -> object:
        return items[0]

    def model_decl(self, meta, items: List[object]) -> ast.ModelDecl:
        name = items[0]
        params = items[1:]
        return ast.ModelDecl(name, params, span=_span(meta))

    def param_decl(self, meta, items: List[object]) -> ast.ParamDecl:
        name = items[0]
        shape = None
        init = None
        for item in items[1:]:
            if isinstance(item, list):
                shape = item
            else:
                init = item
        return ast.ParamDecl(name, shape, init, span=_span(meta))

    def shape_spec(self, meta, items: List[int]) -> List[int]:
        return [int(value) for value in items]

    def init_clause(self, meta, items: List[ast.Expr]) -> ast.Expr:
        return items[0]

    def rule_decl(self, meta, items: List[object]) -> ast.RuleDecl:
        name = items[0]
        idx = 1
        params: List[ast.RuleParam] = []
        if idx < len(items) and isinstance(items[idx], list):
            params = items[idx]
            idx += 1
        return_type = items[idx]
        clauses = items[idx + 1 :]
        return ast.RuleDecl(name, params, return_type, clauses, span=_span(meta))

    def rule_param_list(self, meta, items: List[ast.RuleParam]) -> List[ast.RuleParam]:
        return items

    def rule_param(self, meta, items: List[object]) -> ast.RuleParam:
        name = items[0]
        type_ref = items[1]
        nondiff = False
        if len(items) > 2:
            nondiff = True
        return ast.RuleParam(name, type_ref, nondiff, span=_span(meta))

    def param_attr(self, meta, items: List[object]) -> bool:
        return True

    def rule_forward(self, meta, items: List[ast.Expr]) -> ast.RuleClause:
        return ast.RuleClause("forward", None, items[0], span=_span(meta))

    def rule_deriv(self, meta, items: List[object]) -> ast.RuleClause:
        target = items[0]
        expr = items[1]
        return ast.RuleClause("deriv", target, expr, span=_span(meta))

    def fn_decl(self, meta, items: List[object]) -> ast.FnDecl:
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
        return ast.FnDecl(name, params, return_type, body, span=_span(meta))

    def fn_param_list(self, meta, items: List[ast.FnParam]) -> List[ast.FnParam]:
        return items

    def fn_param(self, meta, items: List[object]) -> ast.FnParam:
        name = items[0]
        type_ref = None
        shape = None
        for item in items[1:]:
            if isinstance(item, ast.TypeRef):
                type_ref = item
            elif isinstance(item, list):
                shape = item
        return ast.FnParam(name, type_ref, shape, span=_span(meta))

    def fn_return(self, meta, items: List[ast.TypeRef]) -> ast.TypeRef:
        return items[0]

    def type_ref(self, meta, items: List[object]) -> ast.TypeRef:
        name = items[0]
        shape = None
        if len(items) > 1:
            shape = items[1]
        return ast.TypeRef(name, shape, span=_span(meta))

    def block(self, meta, items: List[ast.Stmt]) -> ast.Block:
        return ast.Block(items, span=_span(meta))

    def graph_stmt(self, meta, items: List[ast.GraphItem]) -> ast.GraphStmt:
        return ast.GraphStmt(items, span=_span(meta))

    def graph_item(self, meta, items: List[object]) -> ast.GraphItem:
        name = items[0]
        expr = items[1]
        return ast.GraphItem(name, expr, span=_span(meta))

    def assign_stmt(self, meta, items: List[object]) -> ast.AssignStmt:
        return ast.AssignStmt(items[0], items[1], span=_span(meta))

    def grad_stmt(self, meta, items: List[object]) -> ast.GradStmt:
        name = items[0]
        loss_name = items[1]
        params = items[2]
        return ast.GradStmt(name, loss_name, params, span=_span(meta))

    def fetch_stmt(self, meta, items: List[object]) -> ast.FetchStmt:
        target_name = items[0]
        source_name = items[1]
        return ast.FetchStmt(target_name, source_name, span=_span(meta))

    def return_stmt(self, meta, items: List[object]) -> ast.ReturnStmt:
        value = items[0] if items else None
        return ast.ReturnStmt(value, span=_span(meta))

    def loss_stmt(self, meta, items: List[object]) -> ast.LossStmt:
        name = items[0]
        expr = items[1]
        return ast.LossStmt(name, expr, span=_span(meta))

    def ident_list(self, meta, items: List[str]) -> List[str]:
        return items

    def explain_stmt(self, meta, items: List[object]) -> ast.ExplainStmt:
        name = items[0]
        level = None
        output_path = None
        for item in items[1:]:
            if isinstance(item, int):
                level = item
            elif isinstance(item, ast.Expr):
                output_path = item
            elif isinstance(item, str):
                output_path = item
        return ast.ExplainStmt(name, level, span=_span(meta), output_path=output_path)

    def step_stmt(self, meta, items: List[object]) -> ast.StepStmt:
        optimizer = items[0]
        grad_name = items[1]
        return ast.StepStmt(optimizer, grad_name, span=_span(meta))

    def for_stmt(self, meta, items: List[object]) -> ast.ForStmt:
        var_name = items[0]
        start, end = items[1]
        body = items[2]
        return ast.ForStmt(var_name, start, end, body, span=_span(meta))

    def range_expr(self, meta, items: List[ast.Expr]) -> Tuple[ast.Expr, ast.Expr]:
        return items[0], items[1]

    def expr_stmt(self, meta, items: List[ast.Expr]) -> ast.ExprStmt:
        return ast.ExprStmt(items[0], span=_span(meta))

    def ternary(self, meta, items: List[ast.Expr]) -> ast.Expr:
        if len(items) == 1:
            return items[0]
        return ast.TernaryOp(items[0], items[1], items[2], span=_span(meta))

    def compare(self, meta, items: List[object]) -> ast.Expr:
        return _fold_bin(items, _span(meta))

    def sum(self, meta, items: List[object]) -> ast.Expr:
        return _fold_bin(items, _span(meta))

    def term(self, meta, items: List[object]) -> ast.Expr:
        return _fold_bin(items, _span(meta))

    def unary_neg(self, meta, items: List[ast.Expr]) -> ast.UnaryOp:
        return ast.UnaryOp("-", items[0], span=_span(meta))

    def postfix(self, meta, items: List[ast.Expr]) -> ast.Expr:
        expr = items[0]
        for index in items[1:]:
            expr = ast.IndexExpr(expr, index, span=_span(meta))
        return expr

    def number(self, meta, items: List[object]) -> ast.Number:
        value = float(items[0])
        return ast.Number(value, span=_span(meta))

    def string(self, meta, items: List[str]) -> ast.StringLiteral:
        return ast.StringLiteral(items[0], span=_span(meta))

    def list_literal(self, meta, items: List[ast.Expr]) -> ast.ListLiteral:
        return ast.ListLiteral(items, span=_span(meta))

    def name(self, meta, items: List[str]) -> ast.Name:
        return ast.Name(items[0], span=_span(meta))

    def call_expr(self, meta, items: List[object]) -> ast.CallExpr:
        func = items[0]
        args = items[1] if len(items) > 1 else []
        return ast.CallExpr(func, args, span=_span(meta))

    def arg_list(self, meta, items: List[ast.CallArg]) -> List[ast.CallArg]:
        return items

    def arg_pos(self, meta, items: List[ast.Expr]) -> ast.CallArg:
        return ast.CallArg(None, items[0], span=_span(meta))

    def arg_kw(self, meta, items: List[object]) -> ast.CallArg:
        name = items[0]
        value = items[1]
        return ast.CallArg(name, value, span=_span(meta))

    def STRING(self, token: Token) -> str:
        text = str(token)
        if text.startswith('"') or text.startswith("'"):
            return text[1:-1]
        return text

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


def _fold_bin(items: List[object], span: ast.Span) -> ast.Expr:
    if not items:
        raise ValueError("Empty binary expression")
    expr = items[0]
    idx = 1
    while idx < len(items):
        op = str(items[idx])
        right = items[idx + 1]
        expr = ast.BinaryOp(op, expr, right, span=span)
        idx += 2
    return expr
