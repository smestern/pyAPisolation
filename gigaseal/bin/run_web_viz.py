"""Deprecated. Use ``gigaseal web-viz [args...]`` instead."""
import sys
import warnings


def main():
    warnings.warn(
        "run_web_viz.py is deprecated; use `gigaseal web-viz` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from gigaseal.cli import main as cli_main
    raise SystemExit(cli_main(["web-viz", *sys.argv[1:]]))


if __name__ == "__main__":
    main()
