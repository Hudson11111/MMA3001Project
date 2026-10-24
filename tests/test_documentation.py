"""
Documentation tests: every module, public class and public function in the package
has a docstring, and every public function that takes arguments documents them in a
NumPy-style ``Parameters`` section. Keeps the HTML docs (pdoc) complete and in sync.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).parents[1] / "src" / "porkqc"
FILES = sorted(SRC.rglob("*.py"))


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(SRC)))
def test_docstrings_present_and_numpy_style(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert ast.get_docstring(tree), f"{path}: module docstring missing"
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node)
            if not doc:
                problems.append(f"{node.name} (line {node.lineno}): no docstring")
            elif (
                isinstance(node, ast.FunctionDef)
                and not node.name.startswith("_")
                and node.col_offset == 0
                and [a for a in node.args.args if a.arg != "self"]
                and "Parameters" not in doc
            ):
                problems.append(f"{node.name} (line {node.lineno}): no 'Parameters' section")
    assert not problems, f"{path}:\n" + "\n".join(problems)
