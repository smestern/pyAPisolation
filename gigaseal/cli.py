"""
gigaseal unified command-line interface.

Single entry point that exposes every analysis, GUI launcher, and
utility script as a subcommand. Built on top of the modular
``gigaseal.analysis`` framework — any registered ``AnalysisBase``
subclass automatically gets a ``run <module_name>`` subcommand and, if
its parameters are simple types, a dedicated subcommand with auto-built
argparse flags.

Examples
--------
    gigaseal list
    gigaseal run subthreshold data/ --output results/
    gigaseal spike data/demo_data_1.abf --dv-cutoff 10
    gigaseal gui spike-finder
    gigaseal web-viz --data_folder data/
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Importing this package triggers builtin module registration.
from . import analysis as _analysis_pkg  # noqa: F401
from .analysis import (
    AnalysisBase,
    get,
    list_modules,
    get_all,
    run_batch,
    save_results,
)

logger = logging.getLogger(__name__)


# ======================================================================
# GUI launcher registry
# ----------------------------------------------------------------------
# Map subcommand name -> (module_path, callable_name, human description).
# Each launcher is imported lazily so missing optional GUI deps don't
# break ``gigaseal --help``.
# ======================================================================

_GUI_LAUNCHERS: Dict[str, Dict[str, str]] = {
    "spike-finder": {
        "module": "gigaseal.gui.spikeFinder",
        "attr": "main",
        "desc": "Spike detection GUI (legacy spikeFinder)",
    },
    "new-spike-finder": {
        "module": "gigaseal.gui.app",
        "attr": "main",
        "desc": "Next-generation spike finder GUI",
    },
    "database-builder": {
        "module": "gigaseal.gui.databaseBuilder",
        "attr": "run",
        "desc": "Database builder GUI",
    },
    "csv-editor": {
        "module": "gigaseal.gui.csvExcelEditor",
        "attr": "main",
        "desc": "CSV / Excel editor GUI",
    },
    "analysis-wizard": {
        "module": "gigaseal.gui.postAnalysisRunner",
        "attr": "main",
        "desc": "Post-hoc analysis wizard GUI",
    },
    "prism-writer": {
        "module": "gigaseal.dev.prism_writer_gui",
        "attr": "PrismWriterGUI",
        "desc": "Prism (.pzfx) writer GUI",
    },
}


# ======================================================================
# Parser construction
# ======================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gigaseal",
        description="gigaseal — electrophysiology analysis toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase log verbosity (-v INFO, -vv DEBUG)",
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = False  # allow bare `gigaseal` to print help

    # --- list -----------------------------------------------------------
    p_list = sub.add_parser(
        "list",
        help="List all registered analysis modules and their parameters",
    )
    p_list.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON instead of plain text",
    )

    # --- run <module> ---------------------------------------------------
    p_run = sub.add_parser(
        "run",
        help="Run any registered analysis module by name",
    )
    p_run.add_argument("module", help="Registered module name (see `list`)")
    _add_batch_arguments(p_run)
    p_run.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="Override a module parameter (may be repeated)",
    )

    # --- one subcommand per registered module ---------------------------
    # Auto-built argparse flags from each module's `get_parameters()`.
    for name, module in sorted(get_all().items()):
        sub_p = sub.add_parser(
            name,
            help=f"Run {module.display_name}",
            description=(module.__doc__ or "").strip(),
        )
        _add_batch_arguments(sub_p)
        _add_module_parameter_flags(sub_p, module)

    # --- gui <name> -----------------------------------------------------
    p_gui = sub.add_parser("gui", help="Launch one of the bundled GUI apps")
    gui_sub = p_gui.add_subparsers(dest="gui_app", metavar="<app>")
    gui_sub.required = True
    for app_name, info in _GUI_LAUNCHERS.items():
        gui_sub.add_parser(app_name, help=info["desc"])

    # --- web-viz --------------------------------------------------------
    p_web = sub.add_parser(
        "web-viz", help="Launch the Flask web visualization server",
    )
    p_web.add_argument(
        "--backend", type=str, default="dynamic",
        choices=["static", "dynamic"],
        help="Web app backend (default: dynamic)",
    )
    web_grp = p_web.add_mutually_exclusive_group(required=True)
    web_grp.add_argument(
        "--data_folder", type=str,
        help="Folder of ABF files to build a database from",
    )
    web_grp.add_argument(
        "--data_df", type=str,
        help="Path to a pre-generated database file",
    )
    web_grp.add_argument(
        "--data_dir", type=str,
        help="Directory containing a pre-generated database",
    )

    # --- web-analysis ---------------------------------------------------
    p_wa = sub.add_parser(
        "web-analysis",
        help="Launch the interactive analysis web app (Flask)",
    )
    p_wa.add_argument(
        "--profile", choices=["public", "lab"], default=None,
        help="Deployment profile (sets GIGASEAL_WEB_PROFILE). Default: lab.",
    )
    p_wa.add_argument(
        "--host", default="127.0.0.1",
        help="Bind host (default 127.0.0.1; use 0.0.0.0 to expose).",
    )
    p_wa.add_argument("--port", type=int, default=8000, help="Bind port.")
    p_wa.add_argument(
        "--dev", action="store_true",
        help="Run Flask's dev server with debug=True (do not use in production).",
    )

    # --- organize-by-protocol ------------------------------------------
    p_org = sub.add_parser(
        "organize-by-protocol",
        help="Copy ABF files into subfolders named after their protocol",
    )
    p_org.add_argument("source", help="Source directory to scan recursively")
    p_org.add_argument("output", help="Destination root for organized files")
    p_org.add_argument(
        "--dry-run", action="store_true",
        help="Print planned actions without copying",
    )

    # --- convert-config -------------------------------------------------
    p_conv = sub.add_parser(
        "convert-config",
        help="Convert a webViz JSON config to YAML",
    )
    p_conv.add_argument("json_file", help="Input JSON config")
    p_conv.add_argument(
        "-o", "--output", help="Output YAML path (default: <input>.yaml)",
    )
    p_conv.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing YAML file without prompting",
    )

    return parser


def _add_batch_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "input",
        help="ABF file, directory, or glob pattern",
    )
    parser.add_argument(
        "-o", "--output", default=".",
        help="Output directory for results (default: cwd)",
    )
    parser.add_argument(
        "--tag", default="",
        help="Suffix appended to output filenames",
    )
    parser.add_argument(
        "--format", choices=["csv", "xlsx"], default="csv",
        help="Output file format (default: csv)",
    )
    parser.add_argument(
        "--protocol", default=None,
        help="Only process files whose protocol contains this substring",
    )
    parser.add_argument(
        "--sweeps", default=None,
        help="Comma-separated sweep indices (e.g. '0,1,2'); default = all",
    )
    parser.add_argument(
        "--jobs", "-j", type=int, default=1,
        help="Parallel worker processes (default: 1)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="JSON file of parameter overrides (merged before --set / flags)",
    )


def _add_module_parameter_flags(
    parser: argparse.ArgumentParser, module: AnalysisBase,
) -> None:
    """
    Add one argparse flag per parameter declared on *module*.

    Maps:
      * ``bool``        → ``--flag`` / ``--no-flag``
      * ``int``/``float``/``str`` → ``--flag VALUE`` with matching ``type=``
      * anything else  → ``str`` (caller can refine via --set KEY=VALUE)
    """
    for pname, info in module.get_parameters().items():
        cli_flag = "--" + pname.replace("_", "-")
        ptype = info["type"]
        default = info["default"]
        help_str = f"(default: {default!r})"

        if ptype is bool:
            parser.add_argument(
                cli_flag, dest=pname, action="store_true",
                default=None, help=f"Enable {pname} {help_str}",
            )
            parser.add_argument(
                "--no-" + pname.replace("_", "-"),
                dest=pname, action="store_false",
                help=f"Disable {pname}",
            )
        elif ptype in (int, float, str):
            parser.add_argument(
                cli_flag, dest=pname, type=ptype, default=None,
                help=help_str,
            )
        else:
            # Fallback: accept as string, let module coerce later.
            parser.add_argument(
                cli_flag, dest=pname, type=str, default=None,
                help=help_str + " (string-coerced)",
            )


# ======================================================================
# Command handlers
# ======================================================================

def _cmd_list(args: argparse.Namespace) -> int:
    modules = get_all()
    if args.json:
        payload = {}
        for name, mod in modules.items():
            params = {
                pname: {
                    "type": getattr(info["type"], "__name__", str(info["type"])),
                    "default": _jsonable(info["default"]),
                }
                for pname, info in mod.get_parameters().items()
            }
            payload[name] = {
                "display_name": mod.display_name,
                "sweep_mode": mod.sweep_mode,
                "parameters": params,
            }
        print(json.dumps(payload, indent=2, default=str))
        return 0

    if not modules:
        print("No analysis modules registered.")
        return 0

    print(f"{len(modules)} registered analysis module(s):\n")
    for name, mod in sorted(modules.items()):
        print(f"  {name}  —  {mod.display_name}  [{mod.sweep_mode}]")
        params = mod.get_parameters()
        if params:
            for pname, info in params.items():
                tname = getattr(info["type"], "__name__", str(info["type"]))
                print(f"      {pname}: {tname} = {info['default']!r}")
        print()
    return 0


def _cmd_run_module(args: argparse.Namespace, module_name: str) -> int:
    module = get(module_name)
    if module is None:
        print(f"error: unknown module '{module_name}'. "
              f"Use `gigaseal list` to see registered modules.",
              file=sys.stderr)
        return 2

    overrides = _collect_parameter_overrides(args, module)
    if overrides:
        module.set_parameters(**overrides)

    selected_sweeps = _parse_sweeps(args.sweeps)

    logger.info("Running '%s' on %s", module_name, args.input)
    result = run_batch(
        module,
        args.input,
        protocol_filter=args.protocol,
        selected_sweeps=selected_sweeps,
        n_jobs=max(1, args.jobs),
        progress_callback=_print_progress,
    )

    n_files = result.metadata.get("file_count", 1)
    n_errors = len(result.errors)
    print(f"\nProcessed {n_files} file(s); {n_errors} error(s).")
    if result.errors:
        for err in result.errors[:10]:
            print(f"  - {err}", file=sys.stderr)

    df = result.to_dataframe()
    if df.empty:
        print("No results produced; nothing written.")
        return 1 if n_errors else 0

    path = save_results(result, args.output, tag=args.tag, fmt=args.format)
    print(f"Wrote {len(df)} row(s) to {path}")
    return 0


def _cmd_gui(args: argparse.Namespace) -> int:
    info = _GUI_LAUNCHERS.get(args.gui_app)
    if info is None:
        print(f"error: unknown GUI app '{args.gui_app}'", file=sys.stderr)
        return 2
    try:
        mod = importlib.import_module(info["module"])
    except ImportError as exc:
        print(
            f"error: could not import {info['module']}: {exc}\n"
            f"Install the GUI extras with: pip install 'gigaseal[gui]'",
            file=sys.stderr,
        )
        return 1
    target = getattr(mod, info["attr"], None)
    if target is None:
        print(
            f"error: {info['module']} has no attribute '{info['attr']}'",
            file=sys.stderr,
        )
        return 1
    try:
        target()
    except TypeError:
        # Some targets are classes that take no args; instantiate.
        target  # noqa: B018
    return 0


def _cmd_web_viz(args: argparse.Namespace) -> int:
    try:
        from .webViz import run_web_viz as web_runner
    except ImportError as exc:
        print(
            f"error: could not import web viz module: {exc}\n"
            f"Install web extras with: pip install 'gigaseal[web]'",
            file=sys.stderr,
        )
        return 1
    web_runner.run_web_viz(
        args.data_folder,
        database_file=args.data_df,
        backend=args.backend,
    )
    return 0


def _cmd_web_analysis(args: argparse.Namespace) -> int:
    if args.profile:
        os.environ["GIGASEAL_WEB_PROFILE"] = args.profile
    try:
        from .webViz.analysis_web import create_app
    except ImportError as exc:
        print(
            f"error: could not import analysis web app: {exc}\n"
            f"Install web extras with: pip install 'gigaseal[web]'",
            file=sys.stderr,
        )
        return 1
    app = create_app()
    cfg = app.extensions["gigaseal_config"]
    print(
        f"gigaseal analysis web · profile={cfg.profile} · "
        f"http://{args.host}:{args.port}",
        file=sys.stderr,
    )
    if not args.dev:
        print(
            "warning: Flask dev server is single-threaded. "
            "For production, run: "
            "gunicorn -w 2 -b 0.0.0.0:8000 gigaseal.webViz.analysis_web.wsgi:app",
            file=sys.stderr,
        )
    app.run(host=args.host, port=args.port, debug=args.dev, threaded=True)
    return 0


def _cmd_organize_by_protocol(args: argparse.Namespace) -> int:
    import shutil
    try:
        import pyabf
    except ImportError:
        print("error: pyabf is required for organize-by-protocol",
              file=sys.stderr)
        return 1

    source = os.path.abspath(args.source)
    output = os.path.abspath(args.output)
    if not os.path.isdir(source):
        print(f"error: source directory not found: {source}", file=sys.stderr)
        return 2
    if not args.dry_run:
        os.makedirs(output, exist_ok=True)

    copied = 0
    skipped = 0
    for root, _dirs, files in os.walk(source):
        for fname in files:
            if not fname.lower().endswith(".abf"):
                continue
            fp = os.path.join(root, fname)
            try:
                abf = pyabf.ABF(fp, loadData=False)
                proto = abf.protocol or "unknown"
                if "\\" in proto:
                    proto = proto.split("\\")[-1]
                if "/" in proto:
                    proto = proto.split("/")[-1]
            except Exception as exc:  # pyabf raises a broad set of errors
                logger.warning("Skipping %s: %s", fp, exc)
                skipped += 1
                continue

            dest_dir = os.path.join(output, proto)
            if args.dry_run:
                print(f"would copy {fp} -> {dest_dir}")
            else:
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy2(fp, dest_dir)
            copied += 1

    verb = "would copy" if args.dry_run else "copied"
    print(f"{verb} {copied} file(s); skipped {skipped}")
    return 0


def _cmd_convert_config(args: argparse.Namespace) -> int:
    try:
        import yaml
    except ImportError:
        print("error: PyYAML is required (pip install pyyaml)", file=sys.stderr)
        return 1

    src = args.json_file
    if not os.path.exists(src):
        print(f"error: file not found: {src}", file=sys.stderr)
        return 2

    dest = args.output or (os.path.splitext(src)[0] + ".yaml")
    if os.path.exists(dest) and not args.overwrite:
        reply = input(f"{dest} exists. Overwrite? (y/n): ")
        if reply.strip().lower() != "y":
            print("Cancelled.")
            return 0

    with open(src, "r") as f:
        data = json.load(f)
    with open(dest, "w") as f:
        f.write("# gigaseal WebViz Configuration\n")
        f.write(f"# Converted from {os.path.basename(src)}\n\n")
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {dest}")
    return 0


# ======================================================================
# Helpers
# ======================================================================

def _collect_parameter_overrides(
    args: argparse.Namespace, module: AnalysisBase,
) -> Dict[str, Any]:
    """
    Build the override dict in precedence order:
        1. --config JSON file (lowest)
        2. Per-flag CLI overrides from the argparse Namespace
        3. --set KEY=VALUE pairs (highest)
    Only keys recognized by the module are kept.
    """
    overrides: Dict[str, Any] = {}
    known = set(module.get_parameters().keys())

    if getattr(args, "config", None):
        try:
            with open(args.config, "r") as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: ignoring --config {args.config}: {exc}",
                  file=sys.stderr)
        else:
            overrides.update({k: v for k, v in cfg.items() if k in known})

    for key in known:
        val = getattr(args, key, None)
        if val is not None:
            overrides[key] = val

    for pair in getattr(args, "set", []) or []:
        if "=" not in pair:
            print(f"warning: ignoring malformed --set '{pair}'", file=sys.stderr)
            continue
        k, v = pair.split("=", 1)
        k = k.strip()
        if k in known:
            overrides[k] = v  # AnalysisBase.set_parameters does coercion

    return overrides


def _parse_sweeps(sweeps_arg: Optional[str]) -> Optional[List[int]]:
    if not sweeps_arg:
        return None
    out: List[int] = []
    for tok in sweeps_arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            print(f"warning: ignoring non-integer sweep '{tok}'", file=sys.stderr)
    return out or None


def _print_progress(done: int, total: int) -> None:
    bar_len = 40
    filled = int(bar_len * done / max(total, 1))
    bar = "=" * filled + "-" * (bar_len - filled)
    end = "\n" if done >= total else ""
    print(f"\r[{bar}] {done}/{total}", end=end, flush=True)


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


# ======================================================================
# Entry point
# ======================================================================

def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    logging.basicConfig(
        level=log_levels[min(args.verbose, len(log_levels) - 1)],
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "list":
        return _cmd_list(args)
    if args.command == "run":
        return _cmd_run_module(args, args.module)
    if args.command == "gui":
        return _cmd_gui(args)
    if args.command == "web-viz":
        return _cmd_web_viz(args)
    if args.command == "web-analysis":
        return _cmd_web_analysis(args)
    if args.command == "organize-by-protocol":
        return _cmd_organize_by_protocol(args)
    if args.command == "convert-config":
        return _cmd_convert_config(args)

    # Otherwise it must be one of the auto-generated per-module subcommands.
    if get(args.command) is not None:
        return _cmd_run_module(args, args.command)

    parser.error(f"unknown command: {args.command}")
    return 2  # unreachable


if __name__ == "__main__":
    sys.exit(main())
