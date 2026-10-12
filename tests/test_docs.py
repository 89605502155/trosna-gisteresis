"""Примеры из документации должны выполняться и давать указанный результат."""

import doctest
import re
from pathlib import Path

import trosna_gisteresis.core

ROOT = Path(__file__).parents[1]


def test_module_doctests():
    assert doctest.testmod(trosna_gisteresis.core).failed == 0


def test_readme_example_runs_and_prints_documented_output(capsys):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    code = re.search(r"```python\n(.*?)```", readme, re.S).group(1)
    exec(compile(code, "README.md", "exec"), {})
    out = capsys.readouterr().out.splitlines()
    expected = [m.strip() for m in re.findall(r"print\(.*\)[ \t]+#[ \t]*(.+)", code)]
    assert out == expected
