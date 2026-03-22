from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ml.dexbooru_tag_rating_predictor import DexbooruTagRatingPredictor


class _FakeTok:
    def __init__(self, lemma: str) -> None:
        self.lemma_ = lemma


class _FakeDoc:
    def __init__(self, tokens: list[_FakeTok]) -> None:
        self._tokens = tokens

    def __iter__(self):
        return iter(self._tokens)


@pytest.fixture
def fake_nlp() -> MagicMock:
    nlp = MagicMock()

    def _nlp_call(text: str) -> _FakeDoc:
        if "long_hair" in text or "long hair" in text:
            return _FakeDoc([_FakeTok("long"), _FakeTok("hair")])
        return _FakeDoc([_FakeTok("cat"), _FakeTok("dog")])

    nlp.side_effect = _nlp_call
    return nlp


def test_transform_data_underscores_and_whitespace(fake_nlp: MagicMock) -> None:
    with patch("core.base_predictor.load_sklearn_estimator", return_value=MagicMock()):
        predictor = DexbooruTagRatingPredictor(fake_nlp, Path("/tmp/model.skops"))
    out = predictor.transform_data("  long_hair   solo  ")
    assert out == "hair long"
    fake_nlp.assert_called_once_with("long hair solo")


def test_transform_data_sorted_unique_lemmas(fake_nlp: MagicMock) -> None:
    with patch("core.base_predictor.load_sklearn_estimator", return_value=MagicMock()):
        predictor = DexbooruTagRatingPredictor(fake_nlp, Path("/tmp/model.skops"))

    fake_nlp.side_effect = lambda t: _FakeDoc([_FakeTok("dog"), _FakeTok("cat"), _FakeTok("dog")])

    assert predictor.transform_data("animals") == "cat dog"


def test_transform_data_rejects_empty() -> None:
    nlp = MagicMock()
    with patch("core.base_predictor.load_sklearn_estimator", return_value=MagicMock()):
        predictor = DexbooruTagRatingPredictor(nlp, Path("/tmp/model.skops"))
    with pytest.raises(ValueError, match="empty after normalization"):
        predictor.transform_data("   ")


def test_transform_data_rejects_no_lemmas() -> None:
    nlp = MagicMock(return_value=_FakeDoc([_FakeTok("   "), _FakeTok("")]))
    with patch("core.base_predictor.load_sklearn_estimator", return_value=MagicMock()):
        predictor = DexbooruTagRatingPredictor(nlp, Path("/tmp/model.skops"))
    with pytest.raises(ValueError, match="no lemmas"):
        predictor.transform_data("x")
