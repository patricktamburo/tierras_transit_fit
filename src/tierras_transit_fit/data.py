"""Load, clean, and bin light-curve data."""
import logging

import numpy as np
import pandas as pd
from scipy.stats import sigmaclip

from .config import resolve_path

log = logging.getLogger(__name__)


def mask_data(data, mask):
    """Return a new dict with every array cut by `mask` (boolean or index array)."""
    return {k: v[mask] for k, v in data.items()}


def _col(frame, name, what):
    if name not in frame.columns:
        raise KeyError(f"Column '{name}' (for {what}) not found. "
                       f"Available columns: {list(frame.columns)}. "
                       f"Override names in the config under data.lc_columns / "
                       f"data.ancillary_columns.")
    return frame[name].to_numpy(copy=True)


def load_data(cfg):
    d = cfg["data"]
    lc, anc = d["lc_columns"], d["ancillary_columns"]

    df = pd.read_csv(resolve_path(cfg, d["lightcurve"]), comment="#")
    adf = pd.read_csv(resolve_path(cfg, d["ancillary"]))
    if len(adf) != len(df):
        raise ValueError(f"Ancillary file has {len(adf)} rows but light curve has "
                         f"{len(df)}; they must line up row-for-row.")

    return {
        "x": _col(df, lc["time"], "time").astype(float),
        "y": _col(df, lc["flux"], "flux").astype(float),
        "y_err": _col(df, lc["flux_err"], "flux error").astype(float),
        "alc": _col(df, lc["alc"], "ALC").astype(float),
        "alc_err": _col(df, lc["alc_err"], "ALC error").astype(float),
        "sky": _col(df, lc["sky"], "sky").astype(float),
        "wcs_flag": _col(df, lc["wcs_flag"], "WCS flag").astype(bool),
        "pos_flag": _col(df, lc["pos_flag"], "position flag").astype(bool),
        "fwhm_flag": _col(df, lc["fwhm_flag"], "FWHM flag").astype(bool),
        "x_pos": _col(df, lc["x_pos"], "x position").astype(float),
        "y_pos": _col(df, lc["y_pos"], "y position").astype(float),
        "exptime": _col(adf, anc["exptime"], "exposure time").astype(float),
        "airmass": _col(adf, anc["airmass"], "airmass").astype(float),
        "fwhm": _col(adf, anc["fwhm"], "FWHM").astype(float),
    }


def prepare_data(data, sel, dat):
    """Normalize, cut to the transit night, apply quality cuts and sigma clip.

    Returns (data, x_offset); times in the returned data have x_offset subtracted.
    """
    # normalize ALC using the whole dataset
    norm = np.nanmedian(data["alc"])
    data["alc"] = data["alc"] / norm
    data["alc_err"] = data["alc_err"] / norm

    # cut to the transit night
    lo, hi = sel["time_window"]
    data = mask_data(data, (data["x"] > lo) & (data["x"] < hi))
    if len(data["x"]) == 0:
        raise ValueError(f"No data in time_window {sel['time_window']}")

    # normalize flux by the median of the night
    norm2 = np.nanmedian(data["y"])
    data["y"] = data["y"] / norm2
    data["y_err"] = data["y_err"] / norm2

    # quality mask
    quality_mask = dat['quality_mask']

    if quality_mask:
        bad = (data["wcs_flag"] | data["pos_flag"] | data["fwhm_flag"]
            | (data["alc"] < sel["alc_min"])
            | ~np.isfinite(data["y"]) | ~np.isfinite(data["y_err"]))
        data = mask_data(data, ~bad)
    if len(data["x"]) == 0:
        raise ValueError("All points removed by quality cuts.")

    # sigma clip on flux
    sc = sel["sigma_clip"]
    if sc:
        _, l, h = sigmaclip(data["y"], low=sc["low"], high=sc["high"])
        data = mask_data(data, (data["y"] > l) & (data["y"] < h))

    x_offset = int(np.floor(data["x"][0]))
    data["x"] = data["x"] - x_offset
    log.info("%d points retained; time offset = %d", len(data["x"]), x_offset)
    return data, x_offset


def bin_data(data, ppb):
    n = len(data["x"])
    bx, by, bye = [], [], []
    for i0 in range(0, n, ppb):
        sl = slice(i0, min(i0 + ppb, n))  # last bin takes the remainder
        bx.append(np.mean(data["x"][sl]))
        by.append(np.median(data["y"][sl]))
        bye.append(np.median(data["y_err"][sl]) / np.sqrt(len(data["x"][sl])))
    return np.array(bx), np.array(by), np.array(bye)