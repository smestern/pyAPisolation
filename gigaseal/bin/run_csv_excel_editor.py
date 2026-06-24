"""Deprecated. Use ``gigaseal gui csv-editor`` instead."""
import warnings


def main():
    warnings.warn(
        "run_csv_excel_editor.py is deprecated; use "
        "`gigaseal gui csv-editor` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from gigaseal.cli import main as cli_main
    raise SystemExit(cli_main(["gui", "csv-editor"]))


if __name__ == "__main__":
    main()
