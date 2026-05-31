"""Deprecated. Use ``pyapisolation gui spike-finder`` instead.

Kept as a thin shim so existing shortcuts and the ``spike_finder``
console-script entry point in pyproject.toml continue to work.
"""
import multiprocessing
import warnings


def main():
    warnings.warn(
        "run_spike_finder.py is deprecated; use "
        "`pyapisolation gui spike-finder` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from pyAPisolation.cli import main as cli_main
    raise SystemExit(cli_main(["gui", "spike-finder"]))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
