"""End-to-end tests for the analysis web app (Flask).

Covers both deployment profiles:

* ``public`` — upload caps (5 files × 50 MB), no auth.
* ``lab``    — no caps, optional bearer auth.

Uses the bundled ``data/demo_data_1.abf`` for the round-trip test.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import pytest

flask = pytest.importorskip("flask")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_ABF = REPO_ROOT / "data" / "demo_data_1.abf"


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def public_app(tmp_path, monkeypatch):
    monkeypatch.setenv("GIGASEAL_WEB_PROFILE", "public")
    monkeypatch.setenv("GIGASEAL_SESSION_DIR", str(tmp_path))
    monkeypatch.delenv("GIGASEAL_API_TOKEN", raising=False)
    from gigaseal.webViz.analysis_web import create_app, get_config

    return create_app(get_config())


@pytest.fixture
def lab_app(tmp_path, monkeypatch):
    monkeypatch.setenv("GIGASEAL_WEB_PROFILE", "lab")
    monkeypatch.setenv("GIGASEAL_SESSION_DIR", str(tmp_path))
    monkeypatch.delenv("GIGASEAL_API_TOKEN", raising=False)
    from gigaseal.webViz.analysis_web import create_app, get_config

    return create_app(get_config())


@pytest.fixture
def lab_app_with_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("GIGASEAL_WEB_PROFILE", "lab")
    monkeypatch.setenv("GIGASEAL_SESSION_DIR", str(tmp_path))
    monkeypatch.setenv("GIGASEAL_API_TOKEN", "secret-token")
    from gigaseal.webViz.analysis_web import create_app, get_config

    return create_app(get_config())


def _upload(client, name: str, data: bytes):
    return client.post(
        "/api/files",
        data={"files": (io.BytesIO(data), name)},
        content_type="multipart/form-data",
    )


# ----------------------------------------------------------------------
# config + modules
# ----------------------------------------------------------------------


def test_public_config_caps(public_app):
    client = public_app.test_client()
    r = client.get("/api/config")
    assert r.status_code == 200
    body = r.get_json()
    assert body["profile"] == "public"
    assert body["max_files_per_session"] == 5
    assert body["max_file_size_mb"] == 50


def test_lab_config_no_caps(lab_app):
    client = lab_app.test_client()
    body = client.get("/api/config").get_json()
    assert body["profile"] == "lab"
    assert body["max_files_per_session"] is None
    assert body["max_file_size_mb"] is None


def test_modules_endpoint_lists_builtins(public_app):
    client = public_app.test_client()
    body = client.get("/api/modules").get_json()
    # At least one registered analysis must be exposed.
    assert isinstance(body, dict) and body
    sample = next(iter(body.values()))
    assert "name" in sample and "parameters" in sample
    for p in sample["parameters"]:
        assert {"name", "type", "default"} <= p.keys()


# ----------------------------------------------------------------------
# upload + quota
# ----------------------------------------------------------------------


def test_public_rejects_oversized_upload(public_app):
    client = public_app.test_client()
    # Use a buffer just above 50 MB.
    payload = b"x" * (51 * 1024 * 1024)
    r = _upload(client, "big.abf", payload)
    assert r.status_code in (400, 413)
    assert b"limit" in r.data or b"exceed" in r.data.lower()


def test_public_rejects_sixth_file(public_app):
    client = public_app.test_client()
    for i in range(5):
        r = _upload(client, f"sweep_{i}.abf", b"FAKE")
        assert r.status_code == 200, r.data
    r = _upload(client, "extra.abf", b"FAKE")
    assert r.status_code == 400
    assert b"file cap" in r.data


def test_public_rejects_unsupported_extension(public_app):
    client = public_app.test_client()
    r = _upload(client, "evil.exe", b"MZ")
    assert r.status_code == 415


def test_lab_accepts_many_files(lab_app):
    client = lab_app.test_client()
    for i in range(7):
        r = _upload(client, f"sweep_{i}.abf", b"FAKE")
        assert r.status_code == 200


# ----------------------------------------------------------------------
# auth
# ----------------------------------------------------------------------


def test_lab_auth_rejects_missing_token(lab_app_with_auth):
    client = lab_app_with_auth.test_client()
    r = client.get("/api/config")
    assert r.status_code == 401


def test_lab_auth_accepts_bearer_token(lab_app_with_auth):
    client = lab_app_with_auth.test_client()
    r = client.get(
        "/api/config", headers={"Authorization": "Bearer secret-token"}
    )
    assert r.status_code == 200


# ----------------------------------------------------------------------
# round-trip
# ----------------------------------------------------------------------


@pytest.mark.skipif(not DEMO_ABF.is_file(), reason="bundled demo ABF missing")
def test_round_trip_upload_run_export(lab_app):
    client = lab_app.test_client()

    # Upload the real demo file so loadABF actually has something to parse.
    with open(DEMO_ABF, "rb") as fh:
        r = client.post(
            "/api/files",
            data={"files": (fh, DEMO_ABF.name)},
            content_type="multipart/form-data",
        )
    assert r.status_code == 200, r.data

    # Pick the first registered module.
    modules = client.get("/api/modules").get_json()
    assert modules
    module_name = next(iter(modules))

    submit = client.post(
        "/api/jobs",
        json={"module": module_name, "params": {}, "files": [DEMO_ABF.name]},
    )
    assert submit.status_code == 202, submit.data
    job_id = submit.get_json()["job_id"]

    # Poll for completion (job runs in a thread pool).
    deadline = time.time() + 60
    job = None
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").get_json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.2)
    assert job is not None
    assert job["status"] == "done", job.get("error")
    assert job["row_count"] >= 1

    r = client.get(f"/api/jobs/{job_id}/export.csv")
    assert r.status_code == 200
    assert r.data.splitlines()[0]  # has a header row
