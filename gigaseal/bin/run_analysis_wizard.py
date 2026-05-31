"""Deprecated. Use ``gigaseal gui analysis-wizard`` instead."""
import warnings


def main():
    warnings.warn(
        "run_analysis_wizard.py is deprecated; use "
        "`gigaseal gui analysis-wizard` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from gigaseal.cli import main as cli_main
    raise SystemExit(cli_main(["gui", "analysis-wizard"]))


if __name__ == "__main__":
    main()

