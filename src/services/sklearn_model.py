"""Load pretrained scikit-learn models (skops) for the API.

Training notebooks save artifacts under ``model_training/`` via ``model_training.ml.utils.persistence``;
this module only loads those files.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from skops.io import get_untrusted_types, load


def load_sklearn_estimator(
    path: Path | str,
    *,
    trusted: Sequence[str] | None = None,
) -> Any:
    """Load a ``.skops`` file written from the training notebooks.

    If ``trusted`` is omitted, types returned by :func:`get_untrusted_types` are
    passed to :func:`skops.io.load`. Audit that list when training dependencies
    change; do not load untrusted files without reviewing it.
    """
    path = Path(path)
    extra_trusted = list(trusted) if trusted is not None else get_untrusted_types(file=path)
    return load(path, trusted=extra_trusted or None)
