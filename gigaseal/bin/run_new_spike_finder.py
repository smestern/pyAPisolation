"""Deprecated. Use ``gigaseal gui new-spike-finder`` instead."""
import multiprocessing
import warnings


def main():
    warnings.warn(
        "run_new_spike_finder.py is deprecated; use "
        "`gigaseal gui new-spike-finder` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from gigaseal.cli import main as cli_main
    raise SystemExit(cli_main(["gui", "new-spike-finder"]))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
