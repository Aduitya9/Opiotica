"""Static checks on the generated notebooks, so a broken one cannot reach Colab.

These exist because of a real failure: notebook 01 assumed notebook 00's `/content` was still
there, and died ten lines in with `IndexError: list index out of range` — naming neither the
cause nor the cure, after the user had already waited for pip to build two wheels.

Three classes of mistake are checked here:

1. **a name used before any cell defines it** — the error Colab reports as `NameError`
   halfway down a notebook
2. **a prerequisite read without being guarded** — every GPU notebook must declare what it
   needs through `colab.require` and fetch the release through `colab.ensure_release`
3. **a notebook that is not valid JSON, or whose cells do not parse**

Run:  python -m pytest whose-eyes/tests -q
"""

from __future__ import annotations

import ast
import builtins
import json
import os
import re

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NOTEBOOKS = os.path.join(ROOT, "notebooks")

#: Colab injects these; they are never assigned in a cell.
AMBIENT = {"display", "get_ipython", "In", "Out", "exit", "quit", "__name__", "__file__"}

NAMES = ["00_inventory.ipynb", "01_frozen_localiser.ipynb", "02_noise_floor_gate.ipynb"]


def load(name: str) -> list[str]:
    """The code cells of a notebook, in order, with IPython magics blanked out."""
    with open(os.path.join(NOTEBOOKS, name), encoding="utf-8") as fh:
        nb = json.load(fh)
    cells = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        cells.append("\n".join("" if re.match(r"\s*[!%]", line) else line
                               for line in source.split("\n")))
    return cells


def code_without_comments(text: str) -> str:
    """Drop comments so a check cannot match the prose explaining why something is avoided."""
    lines = []
    for line in text.split("\n"):
        if line.lstrip().startswith("#"):
            lines.append("")
            continue
        lines.append(re.sub(r"\s+#.*$", "", line))
    return "\n".join(lines)


def bound_by(node: ast.AST) -> set[str]:
    """Every name this statement binds, including tuple targets and `as` clauses."""
    names: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
            names.add(sub.id)
        elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(sub.name)
        elif isinstance(sub, (ast.Import, ast.ImportFrom)):
            for alias in sub.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(sub, (ast.ExceptHandler,)) and sub.name:
            names.add(sub.name)
    return names


DEFERRED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def loaded_by(node: ast.AST) -> set[str]:
    """Names this statement reads.

    Bodies of functions, classes and lambdas are skipped: they execute later, so a global
    defined further down the notebook is legitimate inside them. Only their decorators and
    default arguments, which do run now, are inspected.
    """
    names: set[str] = set()

    def walk(current: ast.AST) -> None:
        if isinstance(current, DEFERRED):
            # only what runs now: decorators and default arguments
            for decorator in getattr(current, "decorator_list", []):
                walk(decorator)
            for default in getattr(getattr(current, "args", None), "defaults", []) or []:
                walk(default)
            return
        if isinstance(current, ast.Name) and isinstance(current.ctx, ast.Load):
            names.add(current.id)
        for child in ast.iter_child_nodes(current):
            walk(child)

    if isinstance(node, DEFERRED):
        walk(node)
        return names
    for child in ast.iter_child_nodes(node):
        walk(child)
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        names.add(node.id)
    return names


@pytest.mark.parametrize("name", NAMES)
def test_notebook_is_valid_and_every_cell_parses(name):
    cells = load(name)
    assert cells, f"{name} has no code cells"
    for i, source in enumerate(cells):
        try:
            ast.parse(source)
        except SyntaxError as error:  # pragma: no cover - the message is the point
            pytest.fail(f"{name} cell {i} does not parse: {error}")


def undefined_names(cells: list[str], label: str = "notebook") -> list[str]:
    """Names read before any earlier cell defines them, in execution order."""
    defined = set(dir(builtins)) | AMBIENT
    problems = []
    for index, source in enumerate(cells):
        for statement in ast.parse(source).body:
            # A statement's own bindings - loop targets, comprehension variables, `except as`,
            # imports inside a try - count as available within it. The risk this guards against
            # is a name that NO cell defines, or one defined only in a LATER cell.
            local = bound_by(statement)
            for used in sorted(loaded_by(statement) - defined - local):
                problems.append(f"{label} cell {index}: '{used}' used before assignment")
            defined |= local
    return problems


@pytest.mark.parametrize("name", NAMES)
def test_no_name_is_used_before_a_cell_defines_it(name):
    """Catches the NameError that Colab would otherwise report halfway down the notebook."""
    problems = undefined_names(load(name), name)
    assert not problems, "\n".join(problems)


def test_the_checker_is_not_vacuous():
    """A check that never fires is worse than none - it buys false confidence.

    The three cases below are exactly the mistakes seen in this project; the fourth group
    must stay silent or the checker would be unusable.
    """
    assert undefined_names(["print(RELEASE)"]), "a name no cell defines must be caught"
    assert undefined_names(["print(BATCH)", "BATCH = 8"]), "defined only later must be caught"
    assert undefined_names(["import os", "x = os.path.join(WORK, 'a')"]), \
        "a path built from an undefined root must be caught"

    quiet = [
        "import os, json\nfrom src import colab",                      # imports
        "for seed in (1, 2):\n    print(seed)",                        # loop target
        "rows = [k for k in range(3)]",                                # comprehension
        "def run(label, seed=0):\n    return later_global + label",    # deferred body
        "try:\n    from google.colab import userdata\nexcept Exception as e:\n    print(e)",
        "if True:\n    FLAG = 1\n    if FLAG:\n        pass",          # same-statement binding
    ]
    assert not undefined_names(quiet), "the checker fires on legitimate Python"


def test_comment_stripping_does_not_hide_real_calls():
    """The upload check strips comments; prove it still sees the code underneath."""
    assert "upload_large_folder" not in code_without_comments(
        "# upload_large_folder does not take path_in_repo, so it is not used\nx = 1")
    assert "upload_large_folder" in code_without_comments(
        "hf.upload_large_folder(folder_path=p)  # the mistake")


@pytest.mark.parametrize("name", NAMES[1:])          # the GPU notebooks
def test_gpu_notebooks_preflight_their_prerequisites(name):
    """A notebook may not assume another notebook's VM. It declares what it needs, up front."""
    joined = "\n".join(load(name))
    assert "colab.require_gpu()" in joined, (
        f"{name} must stop early when the runtime has no GPU, not at the first .cuda() call")
    assert "colab.require(" in joined, (
        f"{name} must declare its prerequisites so the error names the notebook that "
        "produces each missing file")
    preflight_at = joined.index("colab.require(")
    for heavy in ("train_one", "smp.Unet", "upload_folder"):
        if heavy in joined:
            assert joined.index(heavy) > preflight_at, (
                f"{name} starts '{heavy}' before the preflight - the point is to fail "
                "before the slow work, not during it")


def test_notebook_01_fetches_the_release_itself():
    """The actual bug: /content is wiped between VMs, so notebook 01 cannot rely on 00."""
    joined = "\n".join(load("01_frozen_localiser.ipynb"))
    assert "colab.ensure_release(RAW)" in joined
    assert "glob.glob(os.path.join(RAW" not in joined, (
        "globbing RAW directly is what failed: an empty folder gives IndexError instead of "
        "a message saying the release needs fetching")


@pytest.mark.parametrize("name", NAMES)
def test_no_notebook_force_upgrades_a_colab_package(name):
    """`pip install -U` on a preinstalled package breaks Colab's own pins.

    Seen for real: `-U huggingface_hub` pulled 2.4.0 and broke `datasets`, which pins
    huggingface-hub<2.0. Nothing needed the newer version.
    """
    with open(os.path.join(NOTEBOOKS, name), encoding="utf-8") as fh:
        source = "".join("".join(c["source"]) for c in json.load(fh)["cells"])
    for line in source.split("\n"):
        if re.match(r"\s*!pip", line):
            assert not re.search(r"(?<!-)\B-U\b|--upgrade", line), (
                f"{name} force-upgrades a package:\n    {line.strip()}\n"
                "Pin it or leave Colab's version alone.")


@pytest.mark.parametrize("name", NAMES)
def test_every_folder_upload_states_where_it_lands(name):
    """`upload_large_folder` has no path_in_repo, so its files land at the repo root.

    That would silently invalidate every path in manifest.csv. Folder uploads must use
    `upload_folder` with an explicit `path_in_repo`.
    """
    # Comments are stripped first: this file explains WHY upload_large_folder is avoided,
    # and a checker that cannot tell an explanation from a call is a checker that cries wolf.
    joined = code_without_comments("\n".join(load(name)))
    assert "upload_large_folder" not in joined, (
        f"{name} uses upload_large_folder, which cannot place files under a prefix")
    for call in re.finditer(r"upload_folder\((.*?)\)\n", joined, re.S):
        assert "path_in_repo" in call.group(1), (
            f"{name} calls upload_folder without path_in_repo")


def test_notebooks_are_generated_from_the_builder():
    """Hand-editing an .ipynb would be silently overwritten by the next build."""
    builder = os.path.join(ROOT, "tools", "build_notebooks.py")
    assert os.path.exists(builder)
    with open(builder, encoding="utf-8") as fh:
        source = fh.read()
    for name in NAMES:
        assert name in source, f"{name} is not produced by tools/build_notebooks.py"
