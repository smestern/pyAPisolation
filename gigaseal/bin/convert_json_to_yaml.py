"""Deprecated. Use ``gigaseal convert-config [args...]`` instead."""
import sys
import warnings


def main():
    warnings.warn(
        "convert_json_to_yaml.py is deprecated; use "
        "`gigaseal convert-config` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from gigaseal.cli import main as cli_main
    raise SystemExit(cli_main(["convert-config", *sys.argv[1:]]))


if __name__ == "__main__":
    main()
