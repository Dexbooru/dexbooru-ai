"""Train a tag-string → rating classifier (hashed binary bag-of-words + MultinomialNB).

Uses HashingVectorizer so the feature space is fixed (no vocabulary table). New text is mapped
into the same buckets forever; incremental training can use ``MultinomialNB.partial_fit`` on
hashed batches (``HashingVectorizer.partial_fit`` is a no-op but keeps the sklearn API).

Run from the repository root, e.g.
``uv run python -m model_training.ml.train_danbooru_tag_string_rating_predictor``.
"""

from __future__ import annotations

import json
import logging
import secrets
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import accuracy_score, classification_report, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from model_training.ml.utils.persistence import (
    MODELS_DIR,
    danbooru_tag_rating_predictor_meta_path,
    danbooru_tag_rating_predictor_path,
    save_sklearn_estimator,
)

logger = logging.getLogger(__name__)

_MODEL_TRAINING_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _MODEL_TRAINING_ROOT / "raw_data" / "danbooru_posts" / "final"

# Fixed hash dimension; keep moderate — MultinomialNB stores dense (n_classes, n_features).
HASHING_N_FEATURES = 2**18

# Idempotent outputs (fixed names; overwritten each successful run)
ALPHA_GRID = np.logspace(-3, 1.0, 12)


def build_dataset(data_dir: Path) -> pd.DataFrame:
    paths = sorted(p for p in data_dir.iterdir() if p.is_file() and p.suffix == ".csv")
    if not paths:
        msg = f"No CSV files found under {data_dir}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    frames: list[pd.DataFrame] = []
    for path in paths:
        logger.info("Reading CSV into dataframe: %s", path.resolve())
        frames.append(pd.read_csv(path))

    df = pd.concat(frames, ignore_index=True)
    logger.info("Combined dataframe shape (rows, cols): %s", df.shape)
    logger.info("Columns: %s", list(df.columns))
    logger.info("Dtypes:\n%s", df.dtypes.to_string())
    na_tag = df["tag_string"].isna().sum()
    if na_tag:
        logger.warning("Dropping %d rows with null tag_string", int(na_tag))
        df = df.dropna(subset=["tag_string"])
    df["tag_string"] = df["tag_string"].astype(str)
    return df


def make_estimator_search(random_state: int) -> GridSearchCV:
    pipeline = Pipeline(
        [
            (
                "vect",
                HashingVectorizer(
                    n_features=HASHING_N_FEATURES,
                    binary=True,
                    alternate_sign=False,
                ),
            ),
            ("clf", MultinomialNB()),
        ]
    )
    param_grid = {"clf__alpha": list(ALPHA_GRID)}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    return GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def save_run_metadata(
    path: Path,
    *,
    random_state: int,
    best_params: dict,
    test_accuracy: float,
    test_recall_macro: float,
    data_dir: Path,
    n_rows: int,
) -> None:
    payload = {
        "saved_at_utc": datetime.now(tz=UTC).isoformat(),
        "random_state": random_state,
        "best_params": _json_safe(best_params),
        "test_accuracy": test_accuracy,
        "test_recall_macro": test_recall_macro,
        "data_dir": str(data_dir.resolve()),
        "n_training_rows_after_concat": n_rows,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote run metadata: %s", path.resolve())


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    random_state = secrets.randbelow(2**31)
    logger.info("Training run random_state=%d (logged for reproducibility)", random_state)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    danbooru_tag_rating_predictor_meta_path().parent.mkdir(parents=True, exist_ok=True)

    try:
        df = build_dataset(DATA_DIR)
    except FileNotFoundError as e:
        logger.error("%s", e)
        raise SystemExit(1) from e

    if df.empty:
        logger.error("Dataset is empty after load; aborting.")
        raise SystemExit(1)

    X = df["tag_string"]
    y = df["rating"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=random_state,
        stratify=y,
    )
    logger.info("Train size=%d, test size=%d (85/15 stratified)", len(X_train), len(X_test))

    search = make_estimator_search(random_state)
    search.fit(X_train, y_train)
    best_params_plain = {k: float(v) for k, v in search.best_params_.items()}
    logger.info("GridSearch best_params=%s, best_cv_f1_macro=%.6f", best_params_plain, search.best_score_)

    best = search.best_estimator_
    y_pred = best.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))

    print(f"test_accuracy={acc:.4f}")
    print(f"test_recall_macro={rec:.4f}")
    logger.info("Test accuracy=%.6f, test recall (macro)=%.6f", acc, rec)
    logger.info("Classification report (test):\n%s", classification_report(y_test, y_pred, zero_division=0))

    model_path = danbooru_tag_rating_predictor_path()
    save_sklearn_estimator(best, model_path)
    logger.info("Saved estimator: %s", model_path.resolve())

    save_run_metadata(
        danbooru_tag_rating_predictor_meta_path(),
        random_state=random_state,
        best_params=best_params_plain,
        test_accuracy=acc,
        test_recall_macro=rec,
        data_dir=DATA_DIR,
        n_rows=len(df),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Training failed: %s", e, exc_info=True)
        raise SystemExit(1) from e
