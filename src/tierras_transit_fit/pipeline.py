"""High-level entry point: config in, results out."""
import json
import logging
import shutil
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

from .config import load_config, resolve_path
from .data import load_data, prepare_data
from .fitting import report, run_mcmc, summarize
from .model import tierras_transit_fit
from .plotting import plot_chains, plot_corner, plot_data

log = logging.getLogger(__name__)


@dataclass
class FitResult:
    fit: tierras_transit_fit
    data: dict
    x_offset: int
    sampler: object = None
    flat_samples: np.ndarray = None
    summary: dict = None
    outdir: object = None
    prefix: str = None
    figures: dict = field(default_factory=dict)


def run(config, *, plot_init=False, nsteps=None, progress=True, show=False):
    """Run the full analysis described by a YAML config file.

    Parameters
    ----------
    config : str or Path
        Path to the YAML config.
    plot_init : bool
        Only plot the data and the initial model; skip the MCMC.
    nsteps : int, optional
        Override ``mcmc.nsteps`` from the config.
    progress : bool
        Show the emcee progress bar.
    show : bool
        Call ``plt.show()`` at the end (otherwise figures are closed after saving).
    """
    cfg = load_config(config)
    if nsteps is not None:
        cfg["mcmc"]["nsteps"] = int(nsteps)

    ppb = cfg["plot"]["points_per_bin"]
    dpi = cfg["plot"]["dpi"]
    ylim = cfg["plot"]["ylim"]

    data = load_data(cfg)
    data, x_offset = prepare_data(data, cfg["selection"])
    fit = tierras_transit_fit(cfg, data, x_offset)

    if plot_init:
        fig = plot_data(data, fit, x_offset, ppb, ylim, theta_init=fit.initial_guess())
        if show:
            plt.show()
        return FitResult(fit=fit, data=data, x_offset=x_offset, figures={"data": fig})

    # ---- MCMC ----
    m = cfg["mcmc"]
    nburn = int(m["burn_frac"] * m["nsteps"])
    sampler = run_mcmc(fit, m["nwalkers"], m["nsteps"], m["seed"], progress)
    flat = sampler.get_chain(discard=nburn, flat=True)
    summary = summarize(flat, fit.free)
    report(sampler, fit, summary, m["nsteps"])

    # ---- outputs ----
    out = cfg["output"]
    outdir = resolve_path(cfg, out["directory"])
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = out["prefix"] or f"{cfg['target']}{cfg['planet']}_{cfg['label']}".rstrip("_")

    with open(outdir / f"{prefix}_summary.json", "w") as f:
        json.dump({"free_parameters": summary, "fixed_parameters": fit.fixed,
                   "x_offset": x_offset, "n_points": int(len(data["x"]))}, f, indent=2)
    np.savez(outdir / f"{prefix}_samples.npz", flat_samples=flat,
             chain=sampler.get_chain(), log_prob=sampler.get_log_prob(),
             names=np.array(fit.free), x_offset=x_offset)
    shutil.copy(cfg["_config_path"], outdir / f"{prefix}_config.yaml")  # provenance

    figures = {
        "data": plot_data(data, fit, x_offset, ppb, ylim, samples=flat),
        "chains": plot_chains(sampler, fit.labels, nburn),
        "corner": plot_corner(flat, fit.labels, fit.truths()),
    }
    if out["save_plots"]:
        for name, fig in figures.items():
            fig.savefig(outdir / f"{prefix}_{name}.png", dpi=dpi)
    if show:
        plt.show()
    else:
        for fig in figures.values():
            plt.close(fig)  # don't leak memory when looping over many targets

    log.info("Results written to %s (prefix: %s)", outdir, prefix)
    return FitResult(fit=fit, data=data, x_offset=x_offset, sampler=sampler,
                     flat_samples=flat, summary=summary, outdir=outdir,
                     prefix=prefix, figures=figures)