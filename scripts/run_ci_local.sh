#!/usr/bin/env bash
# ==============================================================================
# Local CI Runner: Mirrors GitHub Actions CI pipeline locally.
# Runs: Ruff Linting -> MuJoCo Asset Compilation -> Pytest Suite (29 tests)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

# Detect Python interpreter (prefer project venv if present)
if [ -x "$REPO_ROOT/bin/python" ]; then
    PYTHON="$REPO_ROOT/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    echo "ERROR: No suitable python executable found." >&2
    exit 1
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

echo "============================================================"
echo "  Starting Local CI Verification"
echo "  Python: $($PYTHON --version) ($PYTHON)"
echo "  Repo:   $REPO_ROOT"
echo "============================================================"

echo ""
echo "[1/4] Running Ruff code hygiene and lint check..."
if $PYTHON -m ruff --version &>/dev/null; then
    $PYTHON -m ruff check "$REPO_ROOT"
    echo ">> [PASS] Ruff lint passed cleanly."
else
    echo ">> [SKIP] Ruff not installed in python environment."
fi

echo ""
echo "[2/4] Verifying MuJoCo model compilation & assets..."
$PYTHON "$REPO_ROOT/tests/test_model_assets.py"
echo ">> [PASS] Model assets compiled successfully."

echo ""
echo "[3/4] Running test suite via Pytest..."
if $PYTHON -m pytest --version &>/dev/null; then
    $PYTHON -m pytest "$REPO_ROOT/tests" -v
else
    echo ">> Running standalone test runners..."
    $PYTHON "$REPO_ROOT/tests/test_cpg.py"
    $PYTHON "$REPO_ROOT/tests/test_obstacle_env.py"
    $PYTHON "$REPO_ROOT/tests/test_render_headless.py"
fi
echo ">> [PASS] Test suite passed."

echo ""
echo "[4/4] Verifying headless offscreen rendering..."
$PYTHON "$REPO_ROOT/tests/test_render_headless.py"
echo ">> [PASS] Offscreen rendering and GIF writer verified."

echo ""
echo "============================================================"
echo "  ALL CI CHECKS PASSED (Ready for Git Push)"
echo "============================================================"
