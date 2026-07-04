# Author: Tom Sapletta · https://tom.sapletta.com
# Part of the ifURI solution.
#
"""Dependency-health gate (dev-machine lane; skips on CI): fails when a locally-developed
dependency of this repo is stale or shadowed in a venv — the \"edits invisible because a
non-editable copy shadows the repo\" class. The failure message contains the exact
`pip install -e ...` fix. See ~/github/local.dev.md."""
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path.home() / "github" / "local.dev.sh"
GITHUB = Path.home() / "github"


@pytest.mark.skipif(not SCRIPT.exists(), reason="local.dev.sh not on this machine (CI)")
def test_local_dev_dependencies_healthy():
    repo = Path(__file__).resolve().parents[1]
    try:
        rel = str(repo.relative_to(GITHUB))
    except ValueError:
        pytest.skip("repo not under ~/github")
    p = subprocess.run(["bash", str(SCRIPT), "--check", rel],
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, "dep-health FAILED — run the fix printed below:\n" + p.stdout + p.stderr
