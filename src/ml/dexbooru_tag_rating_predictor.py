"""Danbooru-style tag string → content rating (skops pipeline + spaCy lemmatization)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

from sklearn.pipeline import Pipeline
from spacy.language import Language

from core.base_predictor import BasePredictor

_WHITESPACE_RE = re.compile(r"\s+")


class DexbooruTagRatingPredictor(BasePredictor):
    def __init__(self, nlp: Language, skops_path: Path) -> None:
        super().__init__(skops_path)

        self._nlp = nlp

    def transform_data(self, tag_string: str) -> str:
        text = tag_string.strip().replace("_", " ")
        text = _WHITESPACE_RE.sub(" ", text).strip()

        if not text:
            raise ValueError("tag_string is empty after normalization")

        doc = self._nlp(text)

        lemmas = sorted({token.lemma_ for token in doc if token.lemma_.strip()})

        if not lemmas:
            raise ValueError("no lemmas extracted from tag_string after spaCy processing")

        return " ".join(lemmas)

    def predict(self, tag_string: str) -> tuple[str, str, dict[str, float]]:
        transformed = self.transform_data(tag_string)

        batch = [transformed]

        pipeline = cast(Pipeline, self._estimator)

        pred = pipeline.predict(batch)[0]
        probas = pipeline.predict_proba(batch)[0]

        clf = pipeline.named_steps["clf"]
        classes = clf.classes_

        percents = {str(c): round(float(p) * 100, 2) for c, p in zip(classes, probas, strict=True)}

        return transformed, str(pred), percents
