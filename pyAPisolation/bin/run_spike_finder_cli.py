"""Deprecated. Use ``pyapisolation spike <folder> [options]`` instead.

The original interactive prompt-based version has been replaced by the
unified CLI subcommand, which accepts the same parameters as flags.
Run ``pyapisolation spike --help`` to see the available options.
"""
import warnings


def main():
    warnings.warn(
        "run_spike_finder_cli.py is deprecated; use "
        "`pyapisolation spike <folder>` instead. "
        "See `pyapisolation spike --help` for flag equivalents of the "
        "old interactive prompts.",
        DeprecationWarning,
        stacklevel=2,
    )
    print(
        "This interactive CLI has been removed. Use the new subcommand:\n"
        "    pyapisolation spike <folder> [--dv-cutoff ...] [--protocol ...] "
        "[--tag ...]\n"
        "Run `pyapisolation spike --help` for all options."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
