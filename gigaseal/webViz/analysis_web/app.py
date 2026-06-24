"""Flask application factory for the analysis web app."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from flask import (
    Flask,
    abort,
    jsonify,
    request,
    send_file,
    session as flask_session,
)
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from .config import WebConfig, get_config
from .jobs import JobManager, JobRejected
from .session import QuotaError, SessionStore
from .traces import load_trace_payload

logger = logging.getLogger(__name__)

_THIS = Path(__file__).resolve().parent
_TEMPLATES = _THIS / "templates"
_STATIC = _THIS / "static"


def create_app(config: Optional[WebConfig] = None) -> Flask:
    """Build a configured Flask app for the analysis web UI."""

    cfg = config or get_config()

    app = Flask(
        __name__,
        template_folder=str(_TEMPLATES),
        static_folder=str(_STATIC),
        static_url_path="/static",
    )
    app.config["SECRET_KEY"] = cfg.secret_key
    # Reject NaN / Infinity in responses — browsers' JSON.parse can't read
    # them, so emitting them silently breaks the polling UI.
    try:
        app.json.allow_nan = False  # Flask >= 2.2
    except AttributeError:  # pragma: no cover — older Flask
        pass
    if cfg.max_upload_bytes is not None:
        # Flask checks this against the Content-Length header before reading
        # the body. Add a small fudge factor for multipart overhead.
        app.config["MAX_CONTENT_LENGTH"] = cfg.max_upload_bytes + (5 * 1024 * 1024)

    sessions = SessionStore(cfg)
    jobs = JobManager(cfg)
    app.extensions["gigaseal_config"] = cfg
    app.extensions["gigaseal_sessions"] = sessions
    app.extensions["gigaseal_jobs"] = jobs

    _register_auth(app, cfg)
    _register_rate_limit(app, cfg)
    _register_error_handlers(app)
    _register_routes(app, cfg, sessions, jobs)

    return app


# ----------------------------------------------------------------------
# Cross-cutting concerns
# ----------------------------------------------------------------------


def _register_auth(app: Flask, cfg: WebConfig) -> None:
    if not cfg.require_auth or not cfg.api_token:
        return

    expected = cfg.api_token

    @app.before_request
    def _check_token():
        # Static assets are public so the login banner can render.
        if request.path.startswith("/static/"):
            return None
        header = request.headers.get("Authorization", "")
        token = header.removeprefix("Bearer ").strip()
        if not token:
            token = request.args.get("token", "")
        if token != expected:
            return jsonify({"error": "authentication required"}), 401
        return None


def _register_rate_limit(app: Flask, cfg: WebConfig) -> None:
    if not cfg.rate_limit:
        return
    try:
        from flask_limiter import Limiter
        from flask_limiter.util import get_remote_address
    except ImportError:  # pragma: no cover — optional dep
        logger.warning(
            "flask-limiter not installed; rate limiting disabled"
            " (install gigaseal[web] to enable)"
        )
        return

    def _key() -> str:
        sid = flask_session.get("sid")
        return sid or get_remote_address() or "anon"

    # No default limits — the JS polls /api/jobs/<id> twice a second, and a
    # global default would 429 the whole UI. Apply the cap only to write
    # endpoints (upload, demo load, job submit) via the .limit() decorator
    # in _register_routes.
    limiter = Limiter(
        app=app,
        key_func=_key,
        storage_uri="memory://",
    )
    app.extensions["gigaseal_limiter"] = (limiter, cfg.rate_limit)


def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(QuotaError)
    def _quota(exc: QuotaError):
        return jsonify({"error": str(exc)}), exc.status_code

    @app.errorhandler(JobRejected)
    def _busy(exc: JobRejected):
        return jsonify({"error": str(exc)}), 429

    @app.errorhandler(RequestEntityTooLarge)
    def _too_big(_exc):
        return jsonify({"error": "upload exceeds size limit"}), 413

    @app.errorhandler(HTTPException)
    def _http(exc: HTTPException):
        return jsonify({"error": exc.description}), exc.code


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------


def _current_sid(sessions: SessionStore) -> str:
    sid = flask_session.get("sid")
    if not sid:
        sid = sessions.new_id()
        flask_session["sid"] = sid
        flask_session.permanent = True
    return sid


def _maybe_limit(app: Flask):
    """Return a decorator that applies the configured rate limit, or a no-op."""

    pair = app.extensions.get("gigaseal_limiter")
    if not pair:
        return lambda fn: fn
    limiter, spec = pair
    return limiter.limit(spec)


def _list_modules() -> Dict[str, Dict[str, Any]]:
    """Snapshot of registered analysis modules for the UI."""

    from gigaseal.analysis import get_all

    out: Dict[str, Dict[str, Any]] = {}
    for name, module in get_all().items():
        if getattr(module, "hidden", False):
            continue
        try:
            raw_params = module.get_parameters()
        except Exception:  # noqa: BLE001
            raw_params = {}
        params = []
        for pname, info in raw_params.items():
            ptype = info.get("type", str)
            params.append(
                {
                    "name": pname,
                    "type": _type_name(ptype),
                    "default": _jsonable(info.get("default")),
                    "value": _jsonable(info.get("value")),
                }
            )
        out[name] = {
            "name": name,
            "display_name": getattr(module, "display_name", "") or name,
            "sweep_mode": getattr(module, "sweep_mode", "per_sweep"),
            "doc": (module.__class__.__doc__ or "").strip().split("\n\n")[0],
            "parameters": params,
        }
    return out


def _type_name(t) -> str:
    if t is bool:
        return "bool"
    if t is int:
        return "int"
    if t is float:
        return "float"
    if t is str:
        return "str"
    return getattr(t, "__name__", "str")


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return str(value)
    except Exception:  # noqa: BLE001
        return None


def _coerce_param(ptype: str, raw: Any) -> Any:
    if raw is None:
        return None
    if ptype == "bool":
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}
    if ptype == "int":
        return int(raw)
    if ptype == "float":
        return float(raw)
    return str(raw)


def _register_routes(
    app: Flask, cfg: WebConfig, sessions: SessionStore, jobs: JobManager
) -> None:

    limit = _maybe_limit(app)

    # ------------------------------------------------------------------
    # UI shell + meta
    # ------------------------------------------------------------------

    @app.get("/")
    def index():
        from flask import render_template

        _current_sid(sessions)  # ensure cookie is set on first paint
        return render_template(
            "index.html",
            profile=cfg.profile,
            max_files=cfg.max_files_per_session,
            max_mb=cfg.max_file_size_mb,
            has_demo=bool(cfg.demo_dataset_path),
            allow_server_paths=cfg.allow_server_paths,
        )

    @app.get("/about")
    def about():
        from flask import render_template

        return render_template("about.html", profile=cfg.profile)

    @app.get("/api/config")
    def api_config():
        return jsonify(
            {
                "profile": cfg.profile,
                "max_files_per_session": cfg.max_files_per_session,
                "max_file_size_mb": cfg.max_file_size_mb,
                "has_demo": bool(cfg.demo_dataset_path),
                "allow_server_paths": cfg.allow_server_paths,
                "max_concurrent_jobs": cfg.max_concurrent_jobs,
            }
        )

    @app.get("/api/modules")
    def api_modules():
        return jsonify(_list_modules())

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    @app.get("/api/files")
    def api_files():
        sid = _current_sid(sessions)
        return jsonify(
            {
                "session_id": sid,
                "files": [f.to_dict() for f in sessions.list_files(sid)],
                "quota": sessions.quota_status(sid),
            }
        )

    @app.post("/api/files")
    @limit
    def api_upload():
        sid = _current_sid(sessions)
        sessions.prune_expired()
        added = []
        files = request.files.getlist("files") or list(request.files.values())
        if not files:
            return jsonify({"error": "no files in upload"}), 400
        for storage in files:
            if not storage or not storage.filename:
                continue
            entry = sessions.add_upload(sid, storage.filename, storage.stream)
            added.append(entry.to_dict())
        return jsonify(
            {
                "added": added,
                "files": [f.to_dict() for f in sessions.list_files(sid)],
                "quota": sessions.quota_status(sid),
            }
        )

    @app.delete("/api/files/<path:name>")
    def api_delete_file(name: str):
        sid = _current_sid(sessions)
        removed = sessions.delete_file(sid, name)
        if not removed:
            abort(404, description="file not found")
        return jsonify({"removed": name})

    @app.post("/api/files/demo")
    @limit
    def api_load_demo():
        if not cfg.demo_dataset_path:
            abort(404, description="demo dataset not configured")
        sid = _current_sid(sessions)
        added = sessions.copy_demo(sid)
        return jsonify(
            {
                "added": [f.to_dict() for f in added],
                "files": [f.to_dict() for f in sessions.list_files(sid)],
                "quota": sessions.quota_status(sid),
            }
        )

    @app.get("/api/trace/<path:name>")
    def api_trace(name: str):
        sid = _current_sid(sessions)
        path = sessions.file_path(sid, name)
        if path is None:
            abort(404, description="file not found")
        try:
            payload = load_trace_payload(str(path))
        except Exception as exc:  # noqa: BLE001
            logger.exception("trace load failed for %s", path)
            return jsonify({"error": f"trace load failed: {exc}"}), 500
        return jsonify(payload)

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    @app.post("/api/jobs")
    @limit
    def api_submit_job():
        sid = _current_sid(sessions)
        body = request.get_json(silent=True) or {}
        module_name = body.get("module")
        if not module_name:
            return jsonify({"error": "missing 'module'"}), 400

        modules = _list_modules()
        if module_name not in modules:
            return jsonify({"error": f"unknown module: {module_name}"}), 400

        # Coerce parameters using the module's declared types.
        type_map = {p["name"]: p["type"] for p in modules[module_name]["parameters"]}
        raw_params = body.get("params") or {}
        params: Dict[str, Any] = {}
        for pname, raw in raw_params.items():
            if pname not in type_map:
                continue
            try:
                params[pname] = _coerce_param(type_map[pname], raw)
            except (TypeError, ValueError) as exc:
                return jsonify({"error": f"bad value for {pname!r}: {exc}"}), 400

        requested_files = body.get("files") or []
        if not requested_files:
            requested_files = [f.name for f in sessions.list_files(sid)]
        if not requested_files:
            return jsonify({"error": "no files selected for analysis"}), 400

        resolved: list[str] = []
        for name in requested_files:
            path = sessions.file_path(sid, name)
            if path is None:
                return jsonify({"error": f"file not found: {name}"}), 404
            resolved.append(str(path))

        selected_sweeps = body.get("selected_sweeps")
        if selected_sweeps is not None:
            try:
                selected_sweeps = [int(s) for s in selected_sweeps]
            except (TypeError, ValueError):
                return jsonify({"error": "selected_sweeps must be ints"}), 400

        result_dir = sessions.dir_for(sid, create=True) / "results"
        job = jobs.submit(
            session_id=sid,
            module_name=module_name,
            file_paths=resolved,
            params=params,
            selected_sweeps=selected_sweeps,
            result_dir=result_dir,
        )
        return jsonify(job.to_dict()), 202

    @app.get("/api/jobs/<job_id>")
    def api_job_status(job_id: str):
        sid = _current_sid(sessions)
        job = jobs.get(sid, job_id)
        if job is None:
            abort(404, description="job not found")
        return jsonify(job.to_dict())

    @app.get("/api/jobs/<job_id>/export.<fmt>")
    def api_job_export(job_id: str, fmt: str):
        sid = _current_sid(sessions)
        job = jobs.get(sid, job_id)
        if job is None or job.status != "done" or job.result_path is None:
            abort(404, description="result not available")
        if fmt == "csv":
            return send_file(
                job.result_path,
                mimetype="text/csv",
                as_attachment=True,
                download_name=f"gigaseal_{job.module_name}_{job_id}.csv",
            )
        if fmt == "xlsx":
            import io

            import pandas as pd

            df = pd.read_csv(job.result_path)
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="results")
            buf.seek(0)
            return send_file(
                buf,
                mimetype=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                as_attachment=True,
                download_name=f"gigaseal_{job.module_name}_{job_id}.xlsx",
            )
        abort(400, description=f"unsupported format: {fmt}")
