"""Pytest configuration. Ensures src is on the path for imports."""

import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))


def pytest_configure(config) -> None:
    """Ensure required ML env vars exist before ApplicationSettings is first loaded."""
    skops = root / "models" / "danbooru_tag_rating_predictor.skops"
    os.environ.setdefault("DANBOORU_TAG_RATING_SKOPS_PATH", str(skops))
    import utils.config as config_module

    reset = getattr(config_module, "reset_settings_cache", None)
    if reset is not None:
        reset()
    else:
        setattr(config_module, "_instance", None)
