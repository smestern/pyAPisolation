"""Smoke tests for the unified `gigaseal` CLI."""

import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEMO_ABF = os.path.join(REPO_ROOT, "data", "demo_data_1.abf")
HAS_DEMO = os.path.exists(DEMO_ABF)


def _run_cli(*args, check=True):
    """Invoke the CLI through `python -m gigaseal.cli` and capture output."""
    result = subprocess.run(
        [sys.executable, "-m", "gigaseal.cli", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"CLI exited {result.returncode}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


class TestCLIBasics:
    def test_help_runs(self):
        r = _run_cli("--help")
        assert "gigaseal" in r.stdout

    def test_no_args_prints_help(self):
        r = _run_cli()
        assert "command" in r.stdout.lower()

    def test_list_text(self):
        r = _run_cli("list")
        assert "spike" in r.stdout
        assert "subthreshold" in r.stdout

    def test_list_json(self):
        r = _run_cli("list", "--json")
        payload = json.loads(r.stdout)
        assert "subthreshold" in payload
        assert "parameters" in payload["subthreshold"]

    def test_unknown_module_errors(self):
        r = _run_cli("run", "does_not_exist", "foo", check=False)
        assert r.returncode != 0


class TestCLIRun:
    """End-to-end runs against the bundled demo ABF."""

    pytestmark = pytest.mark.skipif(not HAS_DEMO, reason="demo ABF not present")

    def test_subthreshold_subcommand(self, tmp_path):
        r = _run_cli(
            "subthreshold", DEMO_ABF,
            "-o", str(tmp_path),
            "--tag", "test",
        )
        assert "Wrote" in r.stdout
        produced = list(tmp_path.glob("subthreshold_test.*"))
        assert produced, f"no output file in {tmp_path}"

    def test_generic_run_subcommand(self, tmp_path):
        r = _run_cli(
            "run", "subthreshold", DEMO_ABF,
            "-o", str(tmp_path),
        )
        assert "Wrote" in r.stdout

    def test_parameter_flag_is_accepted(self, tmp_path):
        # `start` is a known float parameter on SubthresholdAnalysis.
        r = _run_cli(
            "subthreshold", DEMO_ABF,
            "-o", str(tmp_path),
            "--start", "0.1",
        )
        assert "Wrote" in r.stdout
