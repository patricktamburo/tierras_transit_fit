"""Diagnostic and result plots."""
import logging

import matplotlib.pyplot as plt
import numpy as np
from corner import corner

from .data import bin_data

log = logging.getLogger(__name__)


def plot_data(data, fit, x_offset, ppb, ylim=None, samples=None, theta_init=None):
    fig, ax = plt.subplots(5, 1, figsize=(10, 10), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1, 1, 1, 1]})
    x = data["x"]
    unique_exp = np.unique(data["exptime"])
    if len(unique_exp) != 1:
        log.warning("More than one exposure time used: %s", unique_exp)
    lab = "/".join(f"{e:g}" for e in unique_exp)

    bx, by, bye = bin_data(data, ppb)
    ax[0].plot(x, data["y"], marker=".", ls="", color="#b0b0b0", label=f"{lab}-s photometry")
    ax[0].errorbar(bx, by, bye, marker="o", color="k", ls="", zorder=3,
                   label=f"{ppb} points per bin")
    if samples is not None:
        rng = np.random.default_rng()
        for ind in rng.integers(len(samples), size=100):
            ax[0].plot(x, fit.model(samples[ind]), color="m", alpha=0.1)
    if theta_init is not None:
        ax[0].plot(x, fit.model(theta_init), color="tab:blue", label="initial model")
    ax[0].set_ylabel("Normalized Flux", fontsize=14)
    ax[0].legend()
    if ylim is None:
        lo, hi = np.percentile(data["y"], [0.5, 99.5])
        pad = 0.2 * (hi - lo)
        ylim = (lo - pad, hi + pad)
    ax[0].set_ylim(*ylim)
    ax[0].set_xlim(x[0], x[-1])

    ax[1].plot(x, data["alc"], marker=".")
    ax[1].set_ylabel("ALC Flux", fontsize=14)
    ax[2].plot(x, data["fwhm"], marker=".", color="tab:orange")
    ax[2].set_ylabel('FWHM (")', fontsize=14)
    ax[3].plot(x, data["sky"], marker=".", color="tab:green")
    ax[3].set_ylabel("Sky Bkg. (ADU/s)", fontsize=14)

    xo, yo = np.median(data["x_pos"]), np.median(data["y_pos"])
    ax[4].plot(x, data["x_pos"] - xo, marker=".", color="tab:red", label=f"X - {xo:.1f}")
    ax[4].plot(x, data["y_pos"] - yo, marker=".", color="tab:purple", label=f"Y - {yo:.1f}")
    ax[4].legend()
    ax[4].set_ylabel("Pixel Position", fontsize=14)

    for a in ax:
        a.tick_params(labelsize=12)
        a.grid(alpha=0.5)
    ax[-1].set_xlabel(f"Time (BJD TDB $- {x_offset}$ )", fontsize=14)
    fig.tight_layout()
    return fig


def plot_chains(sampler, labels, nburn):
    samples = sampler.get_chain()
    ndim = samples.shape[2]
    fig, axes = plt.subplots(ndim, figsize=(10, 2 * ndim + 2), sharex=True)
    axes = np.atleast_1d(axes)
    for i in range(ndim):
        axes[i].plot(samples[:, :, i], "k", alpha=0.3)
        axes[i].set_xlim(0, len(samples))
        axes[i].set_ylabel(labels[i])
        axes[i].yaxis.set_label_coords(-0.1, 0.5)
        axes[i].axvline(nburn, color="r", label="Burn-in")
        if i == 0:
            axes[i].legend()
    axes[-1].set_xlabel("step number")
    fig.tight_layout()
    return fig


def plot_corner(flat, labels, truths):
    return corner(flat, labels=labels, truths=truths)