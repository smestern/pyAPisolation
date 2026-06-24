"""Profile-driven configuration for the analysis web app.

A single ``WebConfig`` dataclass is resolved at app-creation time from
environment variables. Route handlers consult the config; no profile branches
live inside route bodies.
"""

from __future__ import annotations

import os
import secrets
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _env_int(name: str, default: Optional[int]) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class WebConfig:
    """Resolved web-app configuration.

    Built via :func:`get_config` from ``GIGASEAL_WEB_PROFILE`` (``public``
    or ``lab``) and a handful of override env vars.
    """

    profile: str = "lab"

    # Upload limits (None = unlimited; public profile clamps these).
    max_files_per_session: Optional[int] = None
    max_file_size_mb: Optional[int] = None

    # Session lifecycle.
    session_ttl_h: int = 24
    session_root: Path = field(
        default_factory=lambda: Path(tempfile.gettempdir()) / "gigaseal_sessions"
    )

    # Auth (lab profile only).
    require_auth: bool = False
    api_token: Optional[str] = None

    # Server-side file picker (lab profile only).
    allow_server_paths: bool = False
    server_path_root: Optional[Path] = None

    # Rate limiting (public profile only).
    rate_limit: Optional[str] = None  # e.g. "20/minute"

    # Concurrency cap shared across sessions.
    max_concurrent_jobs: int = 4

    # Worker resource ceilings (public profile only, Linux only).
    job_cpu_seconds: Optional[int] = None
    job_memory_mb: Optional[int] = None

    # Demo dataset preload (public profile only).
    demo_dataset_path: Optional[Path] = None

    # Flask secret for signed session cookies.
    secret_key: str = field(default_factory=lambda: secrets.token_hex(32))

    @property
    def max_upload_bytes(self) -> Optional[int]:
        if self.max_file_size_mb is None:
            return None
        return self.max_file_size_mb * 1024 * 1024


def _public_defaults() -> WebConfig:
    repo_data = Path(__file__).resolve().parents[3] / "data"
    return WebConfig(
        profile="public",
        max_files_per_session=5,
        max_file_size_mb=50,
        session_ttl_h=1,
        require_auth=False,
        allow_server_paths=False,
        rate_limit="20/minute",
        max_concurrent_jobs=2,
        job_cpu_seconds=120,
        job_memory_mb=1024,
        demo_dataset_path=repo_data if repo_data.is_dir() else None,
    )


def _lab_defaults() -> WebConfig:
    token = os.environ.get("GIGASEAL_API_TOKEN")
    server_root = os.environ.get("GIGASEAL_SERVER_PATH_ROOT")
    return WebConfig(
        profile="lab",
        max_files_per_session=None,
        max_file_size_mb=None,
        session_ttl_h=24,
        require_auth=bool(token),
        api_token=token,
        allow_server_paths=bool(server_root),
        server_path_root=Path(server_root) if server_root else None,
        rate_limit=None,
        max_concurrent_jobs=_env_int("GIGASEAL_MAX_CONCURRENT_JOBS", 4) or 4,
        demo_dataset_path=None,
    )


def get_config() -> WebConfig:
    """Build a :class:`WebConfig` from environment variables."""

    profile = os.environ.get("GIGASEAL_WEB_PROFILE", "lab").strip().lower()
    if profile == "public":
        cfg = _public_defaults()
    else:
        cfg = _lab_defaults()

    # Generic overrides applied to both profiles.
    session_root = os.environ.get("GIGASEAL_SESSION_DIR")
    if session_root:
        cfg.session_root = Path(session_root)

    ttl = _env_int("GIGASEAL_SESSION_TTL_H", None)
    if ttl is not None:
        cfg.session_ttl_h = ttl

    secret = os.environ.get("GIGASEAL_SECRET_KEY")
    if secret:
        cfg.secret_key = secret

    cfg.session_root.mkdir(parents=True, exist_ok=True)
    return cfg
