"""Transit + baseline model, priors, and probability functions."""
import batman
import numpy as np

from .config import PARAM_LABELS, PARAM_NAMES


class tierras_transit_fit:
    """Holds the data, batman model, priors, and probability functions."""

    def __init__(self, cfg, data, x_offset):
        self.x = data["x"]
        self.y = data["y"]
        self.yerr = data["y_err"]
        self.x_offset = x_offset
        self.x_ref = self.x[0]  # baseline: m*(x - x_ref) + b

        self.specs = {n: cfg["parameters"][n] for n in PARAM_NAMES}
        self.fixed = {n: float(s["fixed"]) for n, s in self.specs.items() if "fixed" in s}
        self.free = [n for n in PARAM_NAMES if n not in self.fixed]

        mcfg = cfg["model"]
        self.params = batman.TransitParams()
        self.params.t0 = 0.0
        self.params.per = 1.0
        self.params.rp = 0.1
        self.params.a = 10.0
        self.params.inc = 90.0
        self.params.ecc = mcfg["ecc"]
        self.params.w = mcfg["w"]
        self.params.u = list(mcfg["u"])
        self.params.limb_dark = mcfg["limb_dark"]
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
        if "init" in s:
            return s["init"]
        return s["mu"], s["sigma"]

    def initial_guess(self):
        return np.array([self._init_dist(n)[0] for n in self.free])

    def truths(self):
        """Prior means for corner-plot reference lines (NaN for uniform priors)."""
        return [self.specs[n]["mu"] if self.specs[n]["prior"] == "gaussian" else np.nan
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
                    raise RuntimeError("Could not initialize walkers inside the prior.")
        return pos

    # -- model -----------------------------------------------------------
    def model(self, theta):
        p = self.full_params(theta)
        self.params.t0 = p["t0"] - self.x_offset
        self.params.per = p["per"]
        self.params.rp = p["rp"]
        self.params.a = p["a"]
        self.params.inc = p["inc"]
        transit = self._bm.light_curve(self.params)
        return transit * (p["m"] * (self.x - self.x_ref) + p["b"])

    # -- probabilities ---------------------------------------------------
    def log_prior(self, theta):
        p = self.full_params(theta)
        if p["per"] <= 0 or p["rp"] <= 0 or p["a"] <= 1 or not (0 <= p["inc"] <= 90):
            return -np.inf
        lp = 0.0
        for name, val in zip(self.free, theta):
            s = self.specs[name]
            if s["prior"] == "gaussian":
                lp += (-0.5 * ((val - s["mu"]) / s["sigma"]) ** 2
                       - np.log(np.sqrt(2 * np.pi) * s["sigma"]))
            elif not (s["min"] <= val <= s["max"]):
                return -np.inf
        return lp

    def log_likelihood(self, theta):
        return -0.5 * np.sum((self.y - self.model(theta)) ** 2 / self.yerr ** 2)

    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)