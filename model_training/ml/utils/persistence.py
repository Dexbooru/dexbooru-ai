"""Save scikit-learn estimators to disk from notebooks (skops). The API only loads these files."""

from pathlib import Path

from sklearn.base import BaseEstimator
from skops.io import dump

_TRAINING_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = _TRAINING_ROOT.parent
ARTIFACTS_DIR = _TRAINING_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_METADATA_DIR = MODELS_DIR / "metadata"

DANBOORU_TAG_RATING_PREDICTOR_BASENAME = "danbooru_tag_rating_predictor.skops"
DANBOORU_TAG_RATING_PREDICTOR_META_BASENAME = "danbooru_tag_rating_predictor.meta.json"


def danbooru_tag_rating_predictor_path() -> Path:
    """Path for the serialized Danbooru tag→rating pipeline (``models/`` at repo root)."""
    return MODELS_DIR / DANBOORU_TAG_RATING_PREDICTOR_BASENAME


def danbooru_tag_rating_predictor_meta_path() -> Path:
    """Path for training-run metadata JSON (``models/metadata/`` at repo root)."""
    return MODELS_METADATA_DIR / DANBOORU_TAG_RATING_PREDICTOR_META_BASENAME


def artifact_path(filename: str = "model.skops") -> Path:
    """Path under ``model_training/artifacts/`` for a serialized model."""
    return ARTIFACTS_DIR / filename


def save_sklearn_estimator(estimator: BaseEstimator, path: Path | str) -> None:
    """Serialize an estimator to a ``.skops`` file (use from notebooks under ``model_training/``)."""
    dump(estimator, path)
