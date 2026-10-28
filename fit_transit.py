#!/usr/bin/env python
"""
Fit a single-night transit with batman + emcee, driven by a YAML config file.

Usage:
    python fit_transit.py config.yaml                 # full fit, save plots/results
    python fit_transit.py config.yaml --plot-init     # just look at data + initial model
    python fit_transit.py config.yaml --interactive   # show figures at the end
    python fit_transit.py config.yaml --nsteps 500    # override config for a quick test
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import batman
import emcee
from scipy.stats import sigmaclip
from corner import corner

PARAM_NAMES = ['t0', 'per', 'rp', 'a', 'inc', 'm', 'b']
PARAM_LABELS = {'t0': 't0', 'per': 'per', 'rp': 'rp/rs', 'a': 'a/rs',
                'inc': 'inc', 'm': 'm', 'b': 'b'}

DEFAULT_LC_COLUMNS = {
    'time': 'BJD TDB', 'flux': 'Flux', 'flux_err': 'Flux Error',
    'alc': 'ALC', 'alc_err': 'ALC Error', 'sky': 'Sky Background (ADU/s)',
    'wcs_flag': 'WCS Flag', 'pos_flag': 'Position Flag',
    'fwhm_flag': 'FWHM Flag', 'x_pos': 'X', 'y_pos': 'Y',
}
DEFAULT_ANC_COLUMNS = {'exptime': 'Exposure Time', 'airmass': 'Airmass',
                       'fwhm': 'FWHM X'}


# --------------------------------------------------------------------------
# Data handling
# --------------------------------------------------------------------------
def mask_data(data, mask):
    """Return a new dict with every array cut by `mask` (boolean or index array)."""
    return {k: v[mask] for k, v in data.items()}


def load_data(cfg, resolve):
    d = cfg['data']
    lc_cols = {**DEFAULT_LC_COLUMNS, **d.get('lc_columns', {})}
    anc_cols = {**DEFAULT_ANC_COLUMNS, **d.get('ancillary_columns', {})}

    df = pd.read_csv(resolve(d['lightcurve']), comment='#')
    anc = pd.read_csv(resolve(d['ancillary']))
    if len(anc) != len(df):
        raise ValueError(f'Ancillary file has {len(anc)} rows but light curve has '
                         f'{len(df)}; they must line up row-for-row.')

    def col(frame, name):
        return frame[name].to_numpy(copy=True)

    data = {
        'x': col(df, lc_cols['time']).astype(float),
        'y': col(df, lc_cols['flux']).astype(float),
        'y_err': col(df, lc_cols['flux_err']).astype(float),
        'alc': col(df, lc_cols['alc']).astype(float),
        'alc_err': col(df, lc_cols['alc_err']).astype(float),
        'sky': col(df, lc_cols['sky']),
        'wcs_flag': col(df, lc_cols['wcs_flag']).astype(bool),
        'pos_flag': col(df, lc_cols['pos_flag']).astype(bool),
        'fwhm_flag': col(df, lc_cols['fwhm_flag']).astype(bool),
        'x_pos': col(df, lc_cols['x_pos']),
        'y_pos': col(df, lc_cols['y_pos']),
        'exptime': col(anc, anc_cols['exptime']),
        'airmass': col(anc, anc_cols['airmass']),
        'fwhm': col(anc, anc_cols['fwhm']),
    }
    return data


def prepare_data(data, sel):
    """Normalize, cut to the transit night, apply quality cuts and sigma clip.
    Returns (data, x_offset). Times in returned data have x_offset subtracted."""
    # normalize ALC using the whole dataset
    norm = np.nanmedian(data['alc'])
    data['alc'] = data['alc'] / norm
    data['alc_err'] = data['alc_err'] / norm

    # cut to the transit night
    lo, hi = sel['time_window']
    data = mask_data(data, (data['x'] > lo) & (data['x'] < hi))
    if len(data['x']) == 0:
        raise ValueError(f'No data in time_window {sel["time_window"]}')

    # normalize flux by the median of the night
    norm2 = np.nanmedian(data['y'])
    data['y'] = data['y'] / norm2
    data['y_err'] = data['y_err'] / norm2

    # quality mask
    bad = (data['wcs_flag'] | data['pos_flag'] | data['fwhm_flag']
           | (data['alc'] < sel.get('alc_min', 0.5))
           | ~np.isfinite(data['y']) | ~np.isfinite(data['y_err']))
    data = mask_data(data, ~bad)

    # sigma clip on flux
    sc = sel.get('sigma_clip', {'low': 4.0, 'high': 4.0})
    if sc:
        _, l, h = sigmaclip(data['y'], low=sc['low'], high=sc['high'])
        data = mask_data(data, (data['y'] > l) & (data['y'] < h))

    # subtract a time offset to keep numbers small
    x_offset = int(np.floor(data['x'][0]))
    data['x'] = data['x'] - x_offset
    print(f'{len(data["x"])} points retained; time offset = {x_offset}')
    return data, x_offset


def bin_data(data, ppb):
    n = len(data['x'])
    bx, by, bye = [], [], []
    for i0 in range(0, n, ppb):
        sl = slice(i0, min(i0 + ppb, n))   # last bin takes the remainder
        bx.append(np.mean(data['x'][sl]))
        by.append(np.median(data['y'][sl]))
        bye.append(np.median(data['y_err'][sl]) / np.sqrt(len(data['x'][sl])))
    return np.array(bx), np.array(by), np.array(bye)


# --------------------------------------------------------------------------
# Model + priors
# --------------------------------------------------------------------------
class TransitFit:
    """Holds the data, transit model, priors, and probability functions."""

    def __init__(self, cfg, data, x_offset):
        self.x = data['x']
        self.y = data['y']
        self.yerr = data['y_err']
        self.x_offset = x_offset
        self.x_ref = self.x[0]          # baseline: m*(x - x_ref) + b

        pcfg = cfg['parameters']
        missing = [p for p in PARAM_NAMES if p not in pcfg]
        if missing:
            raise ValueError(f'Missing parameters in config: {missing}')

        self.specs = {n: pcfg[n] for n in PARAM_NAMES}
        self.fixed = {n: s['fixed'] for n, s in self.specs.items() if 'fixed' in s}
        self.free = [n for n in PARAM_NAMES if n not in self.fixed]
        for n in self.free:
            if self.specs[n].get('prior') not in ('gaussian', 'uniform'):
                raise ValueError(f"Parameter '{n}': prior must be 'gaussian' or "
                                 f"'uniform' (or give 'fixed').")

        mcfg = cfg.get('model', {})
        self.params = batman.TransitParams()
        self.params.t0 = 0.
        self.params.per = 1.
        self.params.rp = 0.1
        self.params.a = 10.
        self.params.inc = 90.
        self.params.ecc = mcfg.get('ecc', 0.)
        self.params.w = mcfg.get('w', 90.)
        self.params.u = list(mcfg.get('u', [0.1, 0.3]))
        self.params.limb_dark = mcfg.get('limb_dark', 'quadratic')
        self._bm = batman.TransitModel(self.params, self.x)

    # -- helpers ---------------------------------------------------------
    @property
    def ndim(self):
        return len(self.free)

    @property
    def labels(self):
        return [PARAM_LABELS[n] for n in self.free]

    def full_params(self, theta):
        p = dict(self.fixed)
        p.update(zip(self.free, theta))
        return p

    def _init_dist(self, name):
        s = self.specs[name]
        if 'init' in s:
            return s['init']
        if s['prior'] == 'gaussian':
            return s['mu'], s['sigma']
        raise ValueError(f"Parameter '{name}' has a uniform prior; "
                         f"add 'init: [mean, std]' in the config.")

    def initial_guess(self):
        return np.array([self._init_dist(n)[0] for n in self.free])

    def truths(self):
        return [self.specs[n]['mu'] if self.specs[n]['prior'] == 'gaussian' else np.nan
                for n in self.free]

    def initial_positions(self, nwalkers, rng, max_tries=100):
        pos = np.zeros((nwalkers, self.ndim))
        for j, n in enumerate(self.free):
            mu, sig = self._init_dist(n)
            pos[:, j] = rng.normal(mu, sig, nwalkers)
        # redraw any walkers that start outside the prior
        for i in range(nwalkers):
            tries = 0
            while not np.isfinite(self.log_prior(pos[i])):
                for j, n in enumerate(self.free):
                    mu, sig = self._init_dist(n)
                    pos[i, j] = rng.normal(mu, sig)
                tries += 1
                if tries > max_tries:
                    raise RuntimeError('Could not initialize walkers inside the prior.')
        return pos

    # -- model -----------------------------------------------------------
    def model(self, theta):
        p = self.full_params(theta)
        self.params.t0 = p['t0'] - self.x_offset
        self.params.per = p['per']
        self.params.rp = p['rp']
        self.params.a = p['a']
        self.params.inc = p['inc']
        transit = self._bm.light_curve(self.params)
        return transit * (p['m'] * (self.x - self.x_ref) + p['b'])

    # -- probabilities ---------------------------------------------------
    def log_prior(self, theta):
        p = self.full_params(theta)
        # non-physical regions
        if p['per'] <= 0 or p['rp'] <= 0 or p['a'] <= 1 or not (0 <= p['inc'] <= 90):
            return -np.inf
        lp = 0.0
        for name, val in zip(self.free, theta):
            s = self.specs[name]
            if s['prior'] == 'gaussian':
                lp += -0.5 * ((val - s['mu']) / s['sigma']) ** 2 \
                      - np.log(np.sqrt(2 * np.pi) * s['sigma'])
            else:  # uniform
                if not (s.get('min', -np.inf) <= val <= s.get('max', np.inf)):
                    return -np.inf
        return lp

    def log_likelihood(self, theta):
        return -0.5 * np.sum((self.y - self.model(theta)) ** 2 / self.yerr ** 2)

    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot_data(data, fit, x_offset, ppb, ylim=None, samples=None, theta_init=None):
    fig, ax = plt.subplots(5, 1, figsize=(10, 10), sharex=True,
                           gridspec_kw={'height_ratios': [2, 1, 1, 1, 1]})
    x = data['x']
    unique_exp = np.unique(data['exptime'])
    if len(unique_exp) != 1:
        print('WARNING: more than one exposure time used!', unique_exp)
        lab = '/'.join(f'{e:g}' for e in unique_exp)
    else:
        lab = f'{unique_exp[0]:g}'

    bx, by, bye = bin_data(data, ppb)
    ax[0].plot(x, data['y'], marker='.', ls='', color='#b0b0b0', label=f'{lab}-s photometry')
    ax[0].errorbar(bx, by, bye, marker='o', color='k', ls='', zorder=3,
                   label=f'{ppb} points per bin')
    if samples is not None:
        rng = np.random.default_rng()
        for ind in rng.integers(len(samples), size=100):
            ax[0].plot(x, fit.model(samples[ind]), color='m', alpha=0.1)
    if theta_init is not None:
        ax[0].plot(x, fit.model(theta_init), color='tab:blue', label='initial model')
    ax[0].set_ylabel('Normalized Flux', fontsize=14)
    ax[0].legend()
    if ylim is None:
        lo, hi = np.percentile(data['y'], [0.5, 99.5])
        pad = 0.2 * (hi - lo)
        ylim = (lo - pad, hi + pad)
    ax[0].set_ylim(*ylim)
    ax[0].set_xlim(x[0], x[-1])

    ax[1].plot(x, data['alc'], marker='.')
    ax[1].set_ylabel('ALC Flux', fontsize=14)
    ax[2].plot(x, data['fwhm'], marker='.', color='tab:orange')
    ax[2].set_ylabel('FWHM (")', fontsize=14)
    ax[3].plot(x, data['sky'], marker='.', color='tab:green')
    ax[3].set_ylabel('Sky Bkg. (ADU/s)', fontsize=14)

    xo, yo = np.median(data['x_pos']), np.median(data['y_pos'])
    ax[4].plot(x, data['x_pos'] - xo, marker='.', color='tab:red', label=f'X - {xo:.1f}')
    ax[4].plot(x, data['y_pos'] - yo, marker='.', color='tab:purple', label=f'Y - {yo:.1f}')
    ax[4].legend()
    ax[4].set_ylabel('Pixel Position', fontsize=14)

    for a in ax:
        a.tick_params(labelsize=12)
        a.grid(alpha=0.5)
    ax[-1].set_xlabel(f'Time (BJD TDB $- {x_offset}$ )', fontsize=14)
    fig.tight_layout()
    return fig


def plot_chains(sampler, labels, nburn):
    samples = sampler.get_chain()
    ndim = samples.shape[2]
    fig, axes = plt.subplots(ndim, figsize=(10, 2 * ndim + 2), sharex=True)
    axes = np.atleast_1d(axes)
    for i in range(ndim):
        axes[i].plot(samples[:, :, i], 'k', alpha=0.3)
        axes[i].set_xlim(0, len(samples))
        axes[i].set_ylabel(labels[i])
        axes[i].yaxis.set_label_coords(-0.1, 0.5)
        axes[i].axvline(nburn, color='r', label='Burn-in')
        if i == 0:
            axes[i].legend()
    axes[-1].set_xlabel('step number')
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def summarize(flat, names):
    out = {}
    for i, n in enumerate(names):
        lo, med, hi = np.percentile(flat[:, i], [16, 50, 84])
        out[n] = {'median': float(med), 'plus': float(hi - med), 'minus': float(med - lo)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config', help='YAML config file')
    ap.add_argument('--interactive', action='store_true', help='show figures at the end')
    ap.add_argument('--plot-init', action='store_true',
                    help='plot data + initial model and exit (no MCMC)')
    ap.add_argument('--nsteps', type=int, help='override mcmc.nsteps')
    args = ap.parse_args()

    if not (args.interactive or args.plot_init):
        plt.switch_backend('Agg')

    config_path = Path(args.config).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if args.nsteps:
        cfg.setdefault('mcmc', {})['nsteps'] = args.nsteps

    base = config_path.parent
    resolve = lambda p: Path(p) if Path(p).is_absolute() else base / p

    # ---- output naming ----
    out_cfg = cfg.get('output', {})
    outdir = resolve(out_cfg.get('directory', '.'))
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = out_cfg.get('prefix') or f"{cfg['target']}{cfg.get('planet', '')}_{cfg.get('label', '')}".rstrip('_')
    save = out_cfg.get('save_plots', True)
    dpi = cfg.get('plot', {}).get('dpi', 300)
    ppb = cfg.get('plot', {}).get('points_per_bin', 64)
    ylim = cfg.get('plot', {}).get('ylim')

    # ---- data ----
    data = load_data(cfg, resolve)
    data, x_offset = prepare_data(data, cfg['selection'])
    fit = TransitFit(cfg, data, x_offset)

    if args.plot_init:
        plot_data(data, fit, x_offset, ppb, ylim, theta_init=fit.initial_guess())
        plt.show()
        return

    # ---- MCMC ----
    m = cfg.get('mcmc', {})
    nwalkers = m.get('nwalkers', 32)
    nsteps = m.get('nsteps', 2000)
    nburn = int(m.get('burn_frac', 0.1) * nsteps)
    seed = m.get('seed')

    rng = np.random.default_rng(seed)
    pos = fit.initial_positions(nwalkers, rng)
    sampler = emcee.EnsembleSampler(nwalkers, fit.ndim, fit.log_probability)
    if seed is not None:
        sampler.random_state = np.random.RandomState(seed).get_state()
    sampler.run_mcmc(pos, nsteps, progress=True)
    flat = sampler.get_chain(discard=nburn, flat=True)

    # ---- diagnostics / results ----
    print(f'Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}')
    tau = sampler.get_autocorr_time(tol=0)
    print('Autocorrelation times (steps):',
          ', '.join(f'{n}={t:.0f}' for n, t in zip(fit.free, tau)))
    if nsteps < 50 * np.max(tau):
        print('WARNING: chain is shorter than 50 autocorrelation times; '
              'consider increasing nsteps.')

    summary = summarize(flat, fit.free)
    for n, s in summary.items():
        print(f"{n:>4s} = {s['median']:.7g} +{s['plus']:.2g} -{s['minus']:.2g}")

    with open(outdir / f'{prefix}_summary.json', 'w') as f:
        json.dump({'free_parameters': summary, 'fixed_parameters': fit.fixed,
                   'x_offset': x_offset, 'n_points': len(data['x'])}, f, indent=2)
    np.savez(outdir / f'{prefix}_samples.npz', flat_samples=flat,
             chain=sampler.get_chain(), log_prob=sampler.get_log_prob(),
             names=np.array(fit.free), x_offset=x_offset)
    shutil.copy(config_path, outdir / f'{prefix}_config.yaml')   # provenance

    # ---- figures ----
    fig1 = plot_data(data, fit, x_offset, ppb, ylim, samples=flat)
    fig2 = plot_chains(sampler, fit.labels, nburn)
    fig3 = corner(flat, labels=fit.labels, truths=fit.truths())
    if save:
        fig1.savefig(outdir / f'{prefix}_data.png', dpi=dpi)
        fig2.savefig(outdir / f'{prefix}_chains.png', dpi=dpi)
        fig3.savefig(outdir / f'{prefix}_corner.png', dpi=dpi)

    if args.interactive:
        plt.show()


if __name__ == '__main__':
    main()