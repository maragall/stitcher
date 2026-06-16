"""Attribute tracemalloc allocations to the enclosing Python function."""

import ast
import functools
from pathlib import Path
from typing import Tuple


@functools.lru_cache(maxsize=None)
def _func_spans(filename: str) -> Tuple[Tuple[int, int, str], ...]:
    """Return (start_line, end_line, name) for every function in a file."""
    try:
        src = Path(filename).read_text()
        tree = ast.parse(src)
    except (OSError, SyntaxError, ValueError):
        return ()
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno)
            spans.append((node.lineno, end, node.name))
    return tuple(spans)


def function_for(filename: str, lineno: int) -> str:
    """Label "<module-stem>:<func>" for the innermost function covering lineno."""
    stem = Path(filename).stem
    best = None  # (start_line, name) of innermost enclosing function
    for start, end, name in _func_spans(filename):
        if start <= lineno <= end and (best is None or start > best[0]):
            best = (start, name)
    return f"{stem}:{best[1]}" if best else f"{stem}:<module>"
