from __future__ import annotations

import argparse
import ast as pyast
import sys
from pathlib import Path
from typing import Dict, Optional

from .errors import WhiteNNError
from .interpreter import Interpreter, InterpreterError
from .parser import parse_program


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a WhiteNN program.")
    parser.add_argument("file", help="Path to a .wnn file")
    parser.add_argument("--fn", help="Function name to run (optional)")
    parser.add_argument(
        "--arg",
        action="append",
        default=[],
        help="Function argument as name=value (repeatable)",
    )
    args = parser.parse_args(argv)

    try:
        if not args.file.endswith(".wnn"):
            raise ValueError("Expected a .wnn file")
        source = Path(args.file).read_text(encoding="utf-8")
        program = parse_program(source, filename=args.file)
        interpreter = Interpreter(program)
        if args.fn:
            fn_args = _parse_args(args.arg)
            ctx = interpreter.run_function(args.fn, fn_args)
        else:
            ctx = interpreter.run_program()
        for text in ctx.explain_outputs:
            print(text)
        return 0
    except (OSError, InterpreterError, WhiteNNError, ValueError) as exc:
        print(f"Error:\n{exc}", file=sys.stderr)
        return 1


def _parse_args(items: list[str]) -> Dict[str, object]:
    parsed: Dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --arg '{item}', expected name=value")
        name, raw = item.split("=", 1)
        parsed[name] = _parse_value(raw)
    return parsed


def _parse_value(raw: str) -> object:
    try:
        return pyast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


if __name__ == "__main__":
    raise SystemExit(main())
