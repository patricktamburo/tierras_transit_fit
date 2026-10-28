"""Configuration defaults, loading, and validation."""
from copy import deepcopy
from importlib import resources
from pathlib import Path

import yaml

PARAM_NAMES = ["t0", "per", "rp", "a", "inc", "m", "b"]
PARAM_LABELS = {"t0": "t0", "per": "per", "rp": "rp/rs", "a": "a/rs",
                "inc": "inc", "m": "m", "b": "b"}

DEFAULT_LC_COLUMNS = {
    "time": "BJD TDB", "flux": "Flux", "flux_err": "Flux Error",
    "alc": "ALC", "alc_err": "ALC Error", "sky": "Sky Background (ADU/s)",
    "wcs_flag": "WCS Flag", "pos_flag": "Position Flag",
    "fwhm_flag": "FWHM Flag", "x_pos": "X", "y_pos": "Y",
}
DEFAULT_ANC_COLUMNS = {"exptime": "Exposure Time", "airmass": "Airmass",
                       "fwhm": "FWHM X"}

DEFAULTS = {
    "planet": "",
    "label": "",
    "output": {"directory": ".", "prefix": None, "save_plots": True},
    "data": {"lc_columns": DEFAULT_LC_COLUMNS,
             "ancillary_columns": DEFAULT_ANC_COLUMNS},
    "selection": {"alc_min": 0.5, "sigma_clip": {"low": 4.0, "high": 4.0}},
    "model": {"ecc": 0.0, "w": 90.0, "limb_dark": "quadratic", "u": [0.1, 0.3]},
    "mcmc": {"nwalkers": 32, "nsteps": 2000, "burn_frac": 0.1, "seed": None},
    "plot": {"points_per_bin": 64, "ylim": None, "dpi": 300},
}


class ConfigError(ValueError):
    """Raised when the config file is missing something or has a bad value."""


def _merge(default, user):
    out = deepcopy(default)
    for k, v in user.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def template_text():
    """Contents of the example config shipped with the package."""
    return (resources.files("transitfit").joinpath("templates")
            .joinpath("example_config.yaml").read_text())


def resolve_path(cfg, p):
    """Relative paths in the config are relative to the config file."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else cfg["_base_dir"] / p


def load_config(path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    # an empty section (e.g. "plot:" with everything commented out) means "defaults"
    user = {k: v for k, v in user.items()
            if not (v is None and isinstance(DEFAULTS.get(k), dict))}
    cfg = _merge(DEFAULTS, user)
    cfg["_config_path"] = path
    cfg["_base_dir"] = path.parent
    validate(cfg)
    return cfg


def validate(cfg):
    if not cfg.get("target"):
        raise ConfigError("'target' is required.")
    for key in ("lightcurve", "ancillary"):
        if not cfg["data"].get(key):
            raise ConfigError(f"'data.{key}' is required.")

    tw = cfg["selection"].get("time_window")
    if not tw or len(tw) != 2 or tw[0] >= tw[1]:
        raise ConfigError("'selection.time_window' must be [start, end] in BJD TDB.")

    pcfg = cfg.get("parameters")
    if not isinstance(pcfg, dict):
        raise ConfigError("'parameters' section is required.")
    missing = [p for p in PARAM_NAMES if p not in pcfg]
    unknown = [p for p in pcfg if p not in PARAM_NAMES]
    if missing:
        raise ConfigError(f"Missing parameters: {missing}")
    if unknown:
        raise ConfigError(f"Unknown parameters: {unknown} (allowed: {PARAM_NAMES})")

    n_free = 0
    for name in PARAM_NAMES:
        s = pcfg[name]
        if "fixed" in s:
            continue
        n_free += 1
        kind = s.get("prior")
        if kind == "gaussian":
            if "mu" not in s or "sigma" not in s or s["sigma"] <= 0:
                raise ConfigError(f"'{name}': gaussian prior needs mu and sigma > 0.")
        elif kind == "uniform":
            if "min" not in s or "max" not in s or s["min"] >= s["max"]:
                raise ConfigError(f"'{name}': uniform prior needs min < max.")
            if "init" not in s:
                raise ConfigError(f"'{name}': uniform prior needs 'init: [mean, std]'.")
        else:
            raise ConfigError(f"'{name}': set 'prior: gaussian|uniform' or 'fixed: <value>'.")
        if "init" in s and len(s["init"]) != 2:
            raise ConfigError(f"'{name}': init must be [mean, std].")

    nw = cfg["mcmc"]["nwalkers"]
    if nw < 2 * n_free or nw % 2:
        raise ConfigError(f"mcmc.nwalkers must be even and >= 2 x (free params = {n_free}).")