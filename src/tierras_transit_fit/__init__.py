"""Config-driven single-transit fitting with batman + emcee."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tierras_transit_fit")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0+unknown"

from .config import ConfigError, load_config  # noqa: E402
from .pipeline import FitResult, run  # noqa: E402

__all__ = ["run", "FitResult", "load_config", "ConfigError", "__version__"]