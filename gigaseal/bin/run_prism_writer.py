"""Deprecated. Use ``gigaseal gui prism-writer`` instead."""
import warnings


def main():
    warnings.warn(
        "run_prism_writer.py is deprecated; use "
        "`gigaseal gui prism-writer` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from gigaseal.cli import main as cli_main
    raise SystemExit(cli_main(["gui", "prism-writer"]))


if __name__ == "__main__":
    main()
