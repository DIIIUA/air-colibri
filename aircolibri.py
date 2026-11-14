"""AirColibri: `python aircolibri.py <doctor|ask|chat|serve|discover> [options]`."""
import sys

sys.dont_write_bytecode = True

from aircolibri_cli.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
