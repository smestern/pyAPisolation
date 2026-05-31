"""Deprecated. Use ``pyapisolation organize-by-protocol <src> <dst>`` instead.

The original interactive (tkinter) version is gone; the new subcommand
takes ``source`` and ``output`` paths as positional arguments.
"""
import sys
import warnings


def main():
    warnings.warn(
        "org_by_protocol.py is deprecated; use "
        "`pyapisolation organize-by-protocol <src> <dst>` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from pyAPisolation.cli import main as cli_main
    raise SystemExit(cli_main(["organize-by-protocol", *sys.argv[1:]]))


if __name__ == "__main__":
    main()
