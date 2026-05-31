"""Deprecated. Use ``gigaseal spike <folder> [options]`` instead.

The original interactive prompt-based version has been replaced by the
unified CLI subcommand, which accepts the same parameters as flags.
Run ``gigaseal spike --help`` to see the available options.
"""
import warnings


def main():
    warnings.warn(
        "run_spike_finder_cli.py is deprecated; use "
        "`gigaseal spike <folder>` instead. "
        "See `gigaseal spike --help` for flag equivalents of the "
        "old interactive prompts.",
        DeprecationWarning,
        stacklevel=2,
    )
    print(
        "This interactive CLI has been removed. Use the new subcommand:\n"
        "    gigaseal spike <folder> [--dv-cutoff ...] [--protocol ...] "
        "[--tag ...]\n"
        "Run `gigaseal spike --help` for all options."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
