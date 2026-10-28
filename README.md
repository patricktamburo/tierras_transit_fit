# tierras_transit_fit

Config-driven single-night transit fitting (batman + emcee) for Tierras light curves.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # recommended
pip install .            # regular install
pip install -e ".[dev]"  # editable + test tools (for development)
```

## Use

```bash
tierras_transit_fit init my_target.yaml            # write a template config, then edit it
tierras_transit_fit run my_target.yaml --plot-init # sanity-check the data and initial model
tierras_transit_fit run my_target.yaml             # full fit
```

Other options: `--interactive`, `--nsteps N`, `--no-progress`. `python -m tierras_transit_fit ...` works too.

From Python:

```python
import tierras_transit_fit
res = tierras_transit_fit.run("my_target.yaml", show=False)
res.summary["rp"]          # {'median': ..., 'plus': ..., 'minus': ...}
res.flat_samples           # posterior samples, columns = res.fit.free
```

## Outputs (in `output.directory`, named `<prefix>_*`)

`summary.json`, `samples.npz`, `config.yaml` (a copy), `data.png`, `chains.png`, `corner.png`.

See the comments in the template config for all options.

## Tests

```bash
pytest
```