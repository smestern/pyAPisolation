"""Deprecated. Use ``pyapisolation gui database-builder`` instead."""
import warnings


def main():
    warnings.warn(
        "run_builddatabase.py is deprecated; use "
        "`pyapisolation gui database-builder` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from pyAPisolation.cli import main as cli_main
    raise SystemExit(cli_main(["gui", "database-builder"]))


if __name__ == "__main__":
    main()
