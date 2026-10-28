import json

import batman
import numpy as np
import pandas as pd
import pytest
import yaml

from tierras_transit_fit import ConfigError, run

T0, PER, RP, A, INC = 2460000.0, 5.0, 0.1, 12.0, 89.0


def make_case(tmp_path, edit=None):
    rng = np.random.default_rng(1)
    t = np.linspace(T0 - 0.25, T0 + 0.25, 300)

    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = T0, PER, RP, A, INC
    p.ecc, p.w, p.u, p.limb_dark = 0.0, 90.0, [0.1, 0.3], "quadratic"
    flux = 1000.0 * batman.TransitModel(p, t).light_curve(p) + rng.normal(0, 1.0, t.size)

    n = t.size
    pd.DataFrame({
        "BJD TDB": t, "Flux": flux, "Flux Error": np.ones(n),
        "ALC": np.full(n, 1000.0), "ALC Error": np.ones(n),
        "Sky Background (ADU/s)": np.full(n, 100.0),
        "WCS Flag": 0, "Position Flag": 0, "FWHM Flag": 0,
        "X": 1000.0 + rng.normal(0, 0.1, n), "Y": 1000.0 + rng.normal(0, 0.1, n),
    }).to_csv(tmp_path / "lc.csv", index=False)
    pd.DataFrame({"Exposure Time": np.full(n, 30.0), "Airmass": np.full(n, 1.2),
                  "FWHM X": np.full(n, 2.0)}).to_csv(tmp_path / "anc.csv", index=False)

    cfg = {
        "target": "Synth", "planet": "b", "label": "test",
        "output": {"directory": "out"},
        "data": {"lightcurve": "lc.csv", "ancillary": "anc.csv"},
        "selection": {"time_window": [T0 - 0.5, T0 + 0.5]},
        "parameters": {
            "t0": {"prior": "gaussian", "mu": T0 + 0.001, "sigma": 0.003},
            "per": {"prior": "gaussian", "mu": PER, "sigma": 1e-4},
            "rp": {"prior": "gaussian", "mu": RP, "sigma": 0.005},
            "a": {"prior": "gaussian", "mu": A, "sigma": 0.5},
            "inc": {"prior": "gaussian", "mu": INC, "sigma": 0.5},
            "m": {"prior": "uniform", "min": -1, "max": 1, "init": [0.0, 0.001]},
            "b": {"prior": "uniform", "min": 0.5, "max": 1.5, "init": [1.0, 0.001]},
        },
        "mcmc": {"nwalkers": 16, "nsteps": 150, "seed": 1},
        "plot": {"points_per_bin": 20, "dpi": 50},
    }
    if edit:
        edit(cfg)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_end_to_end(tmp_path):
    res = run(make_case(tmp_path), progress=False)
    out = tmp_path / "out"
    for suffix in ["summary.json", "samples.npz", "config.yaml",
                   "data.png", "chains.png", "corner.png"]:
        assert (out / f"Synthb_test_{suffix}").exists(), suffix
    summ = json.loads((out / "Synthb_test_summary.json").read_text())
    assert abs(summ["free_parameters"]["t0"]["median"] - T0) < 0.01
    assert res.fit.ndim == 7


def test_fixed_parameter(tmp_path):
    def edit(cfg):
        cfg["parameters"]["inc"] = {"fixed": 89.0}
    res = run(make_case(tmp_path, edit), progress=False)
    assert res.fit.ndim == 6 and "inc" not in res.summary


def test_model_dips_in_transit(tmp_path):
    res = run(make_case(tmp_path), plot_init=True)
    model = res.fit.model(res.fit.initial_guess())
    assert model.min() < 0.995 and model.max() > 0.999


def test_missing_parameter_raises(tmp_path):
    def edit(cfg):
        del cfg["parameters"]["rp"]
    with pytest.raises(ConfigError):
        run(make_case(tmp_path, edit), progress=False)