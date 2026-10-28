"""Command-line interface:  tierras_transit_fit run|init"""
import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .config import ConfigError, template_text


def build_parser():
    ap = argparse.ArgumentParser(prog="tierras_transit_fit", description=__doc__)
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run a fit from a YAML config")
    r.add_argument("config", help="YAML config file")
    r.add_argument("--plot-init", action="store_true",
                   help="plot data + initial model and exit (no MCMC)")
    r.add_argument("--interactive", action="store_true", help="show figures at the end")
    r.add_argument("--nsteps", type=int, help="override mcmc.nsteps")
    r.add_argument("--no-progress", action="store_true", help="hide the progress bar")

    i = sub.add_parser("init", help="write a template config file")
    i.add_argument("path", nargs="?", default="config.yaml")
    i.add_argument("--force", action="store_true", help="overwrite an existing file")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "init":
        path = Path(args.path)
        if path.exists() and not args.force:
            sys.exit(f"error: {path} exists (use --force to overwrite)")
        path.write_text(template_text())
        print(f"Wrote template config to {path}")
        return 0

    # command == "run"
    import matplotlib.pyplot as plt

    show = args.interactive or args.plot_init
    if not show:
        plt.switch_backend("Agg")  # headless: safe on servers / batch jobs

    from .pipeline import run
    try:
        run(args.config, plot_init=args.plot_init, nsteps=args.nsteps,
            progress=not args.no_progress, show=show)
    except (ConfigError, KeyError, ValueError, FileNotFoundError) as e:
        sys.exit(f"error: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())