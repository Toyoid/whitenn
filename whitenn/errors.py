from __future__ import annotations

from typing import Optional

from .ast import Span


class WhiteNNError(Exception):
    pass


class ParseError(WhiteNNError):
    pass


class SemanticError(WhiteNNError):
    pass


class RuntimeError(WhiteNNError):
    pass


def format_error(
    message: str,
    *,
    span: Optional[Span] = None,
    source: Optional[str] = None,
    filename: Optional[str] = None,
    context: Optional[str] = None,
) -> str:
    if context:
        message = f"{message} ({context})"
    hint = _hint_for_message(message)
    if span is None:
        return _with_hint(message, hint)

    location = f"line {span.line}, col {span.column}"
    if filename:
        location = f"{filename}:{span.line}:{span.column}"
    header = f"{location}: {message}"
    if not source:
        return _with_hint(header, hint)

    lines = source.splitlines()
    if span.line - 1 >= len(lines) or span.line <= 0:
        return header
    line_text = lines[span.line - 1]
    caret = " " * max(span.column - 1, 0) + "^"
    body = f"{header}\n  {line_text}\n  {caret}"
    return _with_hint(body, hint)


def _with_hint(message: str, hint: Optional[str]) -> str:
    if not hint:
        return message
    return f"{message}\n  Hint: {hint}"


def _hint_for_message(message: str) -> Optional[str]:
    msg = message.lower()
    if "loss references non-graph name" in msg:
        return "loss can only use names created inside the most recent graph block."
    if "loss requires a preceding graph" in msg:
        return "place 'loss' after a 'graph { ... }' block."
    if "grad requires a preceding graph" in msg:
        return "place 'grad' after a 'graph { ... }' or 'loss' statement."
    if "step expects an optimizer call" in msg:
        return "use: step SGD(lr=0.1) using g;"
    if "sgd requires lr" in msg:
        return "use: step SGD(lr=0.1) using g;"
    if "unknown grad" in msg:
        return "make sure 'grad' is defined before 'step' or 'explain'."
    if "cannot assign to param" in msg:
        return "params are immutable; use 'step' to update them."
    if "binary op '@' failed" in msg or "matmul" in msg:
        return "check matrix shapes: (a,b) @ (b,c)."
    if "unexpected" in msg and "syntax error" in msg:
        return "check missing tokens like ';' or ')' near the caret."
    return None
