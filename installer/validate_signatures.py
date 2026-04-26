#!/usr/bin/env python3
"""Static signature validator for gui/app.py.

For every call to TileFusion(...), tf.<method>(...), and the worker
constructors, check that the keyword arguments actually exist in the
target's signature. Catches the merge-artifact / refactor-drift class of
bug (e.g. registration_channel vs channel_to_use) before we burn
five minutes of CI time on each platform finding it the slow way.

Usage:
    python installer/validate_signatures.py            # exit 1 on errors

Run from the repo root.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


GUI_PATH = Path("gui/app.py")
TILEFUSION_CORE = Path("src/tilefusion/core.py")
REGISTRATION_PATH = Path("src/tilefusion/registration.py")


def _func_args(fn: ast.FunctionDef) -> set[str]:
    return {a.arg for a in (*fn.args.args, *fn.args.kwonlyargs)} - {"self"}


def _accept_kwargs_for_class(path: Path, class_name: str, method: str = "__init__") -> set[str] | None:
    """Return accepted kw arg names for a class's method, or None if not found."""
    if not path.exists():
        return None
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == method:
                    if any(isinstance(a, ast.arg) and a.arg == "kwargs" for a in fn.args.kwonlyargs):
                        return None  # accepts **kwargs — anything goes
                    if fn.args.kwarg is not None:
                        return None
                    return _func_args(fn)
    return None


def _accept_kwargs_for_method(path: Path, class_name: str, method: str) -> set[str] | None:
    return _accept_kwargs_for_class(path, class_name, method)


def _accept_kwargs_for_function(path: Path, function_name: str) -> set[str] | None:
    if not path.exists():
        return None
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            if node.args.kwarg is not None:
                return None
            return _func_args(node)
    return None


def main() -> int:
    src = GUI_PATH.read_text()
    tree = ast.parse(src)

    # Cache target signatures (None = "we couldn't resolve, skip")
    targets: dict[tuple[str, str], set[str] | None] = {
        # callable form: ("Name", call_kind) -> accepted kwargs
        ("TileFusion", "class"): _accept_kwargs_for_class(TILEFUSION_CORE, "TileFusion"),
        ("optimize_shifts", "method"): _accept_kwargs_for_method(TILEFUSION_CORE, "TileFusion", "optimize_shifts"),
        ("refine_tile_positions_with_cross_correlation", "method"): _accept_kwargs_for_method(TILEFUSION_CORE, "TileFusion", "refine_tile_positions_with_cross_correlation"),
        ("save_pairwise_metrics", "method"): _accept_kwargs_for_method(TILEFUSION_CORE, "TileFusion", "save_pairwise_metrics"),
        ("find_adjacent_pairs", "func"): _accept_kwargs_for_function(REGISTRATION_PATH, "find_adjacent_pairs"),
        # Worker classes live in gui/app.py — read from there
        ("PreviewWorker", "class"): _accept_kwargs_for_class(GUI_PATH, "PreviewWorker"),
        ("FusionWorker", "class"): _accept_kwargs_for_class(GUI_PATH, "FusionWorker"),
        ("BatchFusionWorker", "class"): _accept_kwargs_for_class(GUI_PATH, "BatchFusionWorker"),
        ("FlatfieldWorker", "class"): _accept_kwargs_for_class(GUI_PATH, "FlatfieldWorker"),
    }

    errors: list[str] = []

    def check_call(callee_label: str, accepted: set[str] | None, kwargs: list[str], lineno: int) -> None:
        if accepted is None:
            return
        bad = [k for k in kwargs if k not in accepted]
        if bad:
            errors.append(
                f"  L{lineno}: {callee_label}(...) got unexpected kwarg(s) {bad!r}; accepted={sorted(accepted)}"
            )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        kwargs = [kw.arg for kw in node.keywords if kw.arg is not None]

        # Plain Name calls: TileFusion(...), find_adjacent_pairs(...), Worker(...)
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if (name, "class") in targets:
                check_call(name, targets[(name, "class")], kwargs, node.lineno)
            elif (name, "func") in targets:
                check_call(name, targets[(name, "func")], kwargs, node.lineno)

        # Attribute calls: tf.optimize_shifts(...), self.worker.optimize_shifts(...)
        elif isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if (attr, "method") in targets:
                check_call(f".{attr}", targets[(attr, "method")], kwargs, node.lineno)

    if errors:
        print("Signature mismatches:")
        for e in errors:
            print(e)
        print(f"\n{len(errors)} mismatch(es).")
        return 1
    print("Signatures look OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
