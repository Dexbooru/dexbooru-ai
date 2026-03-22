from pathlib import Path

import numpy as np
import pytest
from model_training.ml.utils.persistence import artifact_path, save_sklearn_estimator
from sklearn.linear_model import LogisticRegression

from services.sklearn_model import load_sklearn_estimator


@pytest.fixture
def tiny_xy() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((30, 4))
    y = (rng.random(30) > 0.5).astype(int)
    return x, y


def test_artifact_path_default_filename() -> None:
    path = artifact_path()
    assert path.name == "model.skops"
    assert path.parent.name == "artifacts"


def test_skops_roundtrip(tmp_path: Path, tiny_xy: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = tiny_xy
    clf = LogisticRegression(max_iter=500).fit(x, y)
    path = tmp_path / "model.skops"
    save_sklearn_estimator(clf, path)
    loaded = load_sklearn_estimator(path)
    assert type(loaded) is LogisticRegression
    assert np.allclose(loaded.predict(x), clf.predict(x))
