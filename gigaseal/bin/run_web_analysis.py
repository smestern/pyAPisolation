"""Deprecated. Use ``gigaseal gui web-analysis`` instead."""
import multiprocessing
import warnings


def main():
    warnings.warn(
        "run_web_analysis.py is deprecated; use "
        "`gigaseal gui web-analysis` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from gigaseal.cli import main as cli_main
    raise SystemExit(cli_main(["web-analysis",]))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
