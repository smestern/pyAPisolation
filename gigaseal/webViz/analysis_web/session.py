"""Per-browser session storage for the analysis web app.

Each browser gets a signed-cookie session id; uploads land in
``<session_root>/<sid>/`` and are pruned by TTL. Quota enforcement
(file count, file size) is delegated to :class:`SessionStore` so route
handlers stay profile-agnostic.
"""

from __future__ import annotations

import logging
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .config import WebConfig

logger = logging.getLogger(__name__)

_ALLOWED_SUFFIXES = {".abf", ".nwb"}
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class QuotaError(Exception):
    """Raised when a session-level limit is exceeded."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class FileEntry:
    name: str
    size_bytes: int
    path: Path

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "size_bytes": self.size_bytes,
            "size_mb": round(self.size_bytes / (1024 * 1024), 2),
        }


def _sanitize_filename(name: str) -> str:
    base = Path(name).name
    cleaned = _SAFE_NAME.sub("_", base).strip("._") or "file"
    return cleaned[:120]


class SessionStore:
    """Filesystem-backed per-session upload store."""

    def __init__(self, config: WebConfig):
        self.config = config
        self.root = Path(config.session_root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Session id lifecycle
    # ------------------------------------------------------------------

    def new_id(self) -> str:
        return uuid.uuid4().hex

    def dir_for(self, sid: str, create: bool = False) -> Path:
        sid = self._validate_sid(sid)
        path = self.root / sid
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _validate_sid(sid: str) -> str:
        if not re.fullmatch(r"[a-f0-9]{16,64}", sid or ""):
            raise QuotaError("invalid session id", status_code=400)
        return sid

    # ------------------------------------------------------------------
    # File ops
    # ------------------------------------------------------------------

    def list_files(self, sid: str) -> List[FileEntry]:
        path = self.dir_for(sid)
        if not path.is_dir():
            return []
        entries = []
        for child in sorted(path.iterdir()):
            if child.is_file() and child.suffix.lower() in _ALLOWED_SUFFIXES:
                entries.append(
                    FileEntry(
                        name=child.name,
                        size_bytes=child.stat().st_size,
                        path=child,
                    )
                )
        return entries

    def quota_status(self, sid: str) -> dict:
        files = self.list_files(sid)
        used_bytes = sum(f.size_bytes for f in files)
        return {
            "files_used": len(files),
            "files_max": self.config.max_files_per_session,
            "bytes_used": used_bytes,
            "bytes_max_per_file": self.config.max_upload_bytes,
            "max_file_size_mb": self.config.max_file_size_mb,
        }

    def add_upload(self, sid: str, filename: str, stream) -> FileEntry:
        """Save an uploaded file to the session dir, enforcing quotas.

        ``stream`` is any file-like object with ``read()``.
        """

        safe_name = _sanitize_filename(filename)
        suffix = Path(safe_name).suffix.lower()
        if suffix not in _ALLOWED_SUFFIXES:
            raise QuotaError(
                f"unsupported file type: {suffix or '(none)'}; "
                f"allowed: {sorted(_ALLOWED_SUFFIXES)}",
                status_code=415,
            )

        existing = self.list_files(sid)
        max_files = self.config.max_files_per_session
        if max_files is not None and len(existing) >= max_files:
            raise QuotaError(
                f"session is at file cap ({max_files}); delete one to upload more",
                status_code=400,
            )

        target_dir = self.dir_for(sid, create=True)
        target = target_dir / safe_name
        # Disambiguate name collisions.
        i = 1
        while target.exists():
            target = target_dir / f"{Path(safe_name).stem}_{i}{suffix}"
            i += 1

        max_bytes = self.config.max_upload_bytes
        written = 0
        chunk_size = 1024 * 1024
        try:
            with open(target, "wb") as fh:
                while True:
                    chunk = stream.read(chunk_size)
                    if not chunk:
                        break
                    written += len(chunk)
                    if max_bytes is not None and written > max_bytes:
                        raise QuotaError(
                            f"file exceeds {self.config.max_file_size_mb} MB limit",
                            status_code=413,
                        )
                    fh.write(chunk)
        except QuotaError:
            target.unlink(missing_ok=True)
            raise
        except Exception:
            target.unlink(missing_ok=True)
            raise

        return FileEntry(name=target.name, size_bytes=written, path=target)

    def delete_file(self, sid: str, filename: str) -> bool:
        target = self.dir_for(sid) / _sanitize_filename(filename)
        if target.is_file():
            target.unlink()
            return True
        return False

    def file_path(self, sid: str, filename: str) -> Optional[Path]:
        target = self.dir_for(sid) / _sanitize_filename(filename)
        return target if target.is_file() else None

    def copy_demo(self, sid: str) -> List[FileEntry]:
        """Copy the bundled demo dataset into the session dir (public profile)."""

        src = self.config.demo_dataset_path
        if src is None or not Path(src).is_dir():
            return []
        added: List[FileEntry] = []
        max_files = self.config.max_files_per_session
        for child in sorted(Path(src).iterdir()):
            if max_files is not None and len(self.list_files(sid)) >= max_files:
                break
            if child.is_file() and child.suffix.lower() in _ALLOWED_SUFFIXES:
                with open(child, "rb") as fh:
                    try:
                        added.append(self.add_upload(sid, child.name, fh))
                    except QuotaError:
                        break
        return added

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def prune_expired(self) -> int:
        ttl_seconds = self.config.session_ttl_h * 3600
        cutoff = time.time() - ttl_seconds
        removed = 0
        if not self.root.is_dir():
            return 0
        for child in self.root.iterdir():
            if not child.is_dir():
                continue
            try:
                mtime = max(
                    (p.stat().st_mtime for p in child.rglob("*")),
                    default=child.stat().st_mtime,
                )
            except OSError:
                continue
            if mtime < cutoff:
                try:
                    shutil.rmtree(child, ignore_errors=True)
                    removed += 1
                except OSError as exc:
                    logger.warning("failed to prune session %s: %s", child, exc)
        return removed
