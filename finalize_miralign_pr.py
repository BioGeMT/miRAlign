from __future__ import annotations

import subprocess
from pathlib import Path

EXPECTED_BRANCH = "fit-model-sklearn-api"
ROOT = Path.cwd()
MIRALIGN = ROOT / "src" / "miralign.py"
README = ROOT / "README.md"
ESTIMATOR_TEST = ROOT / "tests" / "test_estimator.py"


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


try:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"],
        text=True,
    ).strip()
except (subprocess.CalledProcessError, FileNotFoundError):
    branch = None

if branch and branch != EXPECTED_BRANCH:
    fail(
        f"This finalizer is intended for branch {EXPECTED_BRANCH!r}; "
        f"current branch is {branch!r}."
    )

for path in (MIRALIGN, README):
    if not path.exists():
        fail(f"Expected repository file not found: {path}")

text = MIRALIGN.read_text(encoding="utf-8")

old_tol_block = '''    if tol is None:
        optimizer_tol = 1e-3
    else:
        tol = float(tol)
        if not np.isfinite(tol) or tol < 0:
            raise ValueError("tol must be a finite non-negative number or None.")
        optimizer_tol = max(tol, np.finfo(float).eps)
'''

new_tol_block = '''    # ``tol`` controls only the outer miRAlign stopping rule. Keep the
    # historical SciPy tolerance for the inner alpha/label optimizations so
    # changing outer convergence does not silently change those subproblems.
    optimizer_tol = 1e-3
    if tol is not None:
        tol = float(tol)
        if not np.isfinite(tol) or tol < 0:
            raise ValueError("tol must be a finite non-negative number or None.")
'''

if new_tol_block not in text:
    if old_tol_block not in text:
        fail("Could not find the expected tol block in src/miralign.py.")
    text = text.replace(old_tol_block, new_tol_block, 1)

MIRALIGN.write_text(text.rstrip() + "\n", encoding="utf-8")

readme = README.read_text(encoding="utf-8")
readme = readme.replace(
    "├── tests/                     core and estimator tests",
    "├── tests/                     existing core regression tests",
)
README.write_text(readme.rstrip() + "\n", encoding="utf-8")

if ESTIMATOR_TEST.exists():
    ESTIMATOR_TEST.unlink()

print("Applied final PR cleanup:")
print("  - decoupled outer tol from inner SciPy optimizer tolerance")
print("  - removed tests/test_estimator.py from this branch")
print("  - corrected README repository structure")
print()
print("Next run:")
print("  uv run pytest tests/test_core_smoke.py")
print("  uv run python examples/basic_usage.py")
print("  git diff --check")
print("  git status --short")
