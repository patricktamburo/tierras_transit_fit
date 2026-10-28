"""emcee wrapper and result summaries."""
import logging

import emcee
import numpy as np

log = logging.getLogger(__name__)


def run_mcmc(fit, nwalkers, nsteps, seed=None, progress=True):
    rng = np.random.default_rng(seed)
    pos = fit.initial_positions(nwalkers, rng)
    sampler = emcee.EnsembleSampler(nwalkers, fit.ndim, fit.log_probability)
    if seed is not None:
        sampler.random_state = np.random.RandomState(seed).get_state()
    sampler.run_mcmc(pos, nsteps, progress=progress)
    return sampler


def summarize(flat, names):
    out = {}
    for i, n in enumerate(names):
        lo, med, hi = np.percentile(flat[:, i], [16, 50, 84])
        out[n] = {"median": float(med), "plus": float(hi - med), "minus": float(med - lo)}
    return out


def report(sampler, fit, summary, nsteps):
    log.info("Mean acceptance fraction: %.3f", np.mean(sampler.acceptance_fraction))
    tau = sampler.get_autocorr_time(tol=0)
    log.info("Autocorrelation times (steps): %s",
             ", ".join(f"{n}={t:.0f}" for n, t in zip(fit.free, tau)))
    if nsteps < 50 * np.max(tau):
        log.warning("Chain is shorter than 50 autocorrelation times; "
                    "consider increasing mcmc.nsteps.")
    for n, s in summary.items():
        log.info("%4s = %.7g +%.2g -%.2g", n, s["median"], s["plus"], s["minus"])