import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Avoid loading the real spaCy model when collecting tests.
def _fake_nlp() -> MagicMock:
    nlp = MagicMock()

    def pipe(texts, **kwargs):
        class Doc:
            def __init__(self, text: str) -> None:
                self._parts = (text or "").split()

            def __iter__(self):
                for w in self._parts:
                    t = MagicMock()
                    t.lemma_ = w
                    yield t

        for text in texts:
            yield Doc(text)

    nlp.pipe = pipe
    return nlp


with patch("spacy.load", return_value=_fake_nlp()):
    import model_training.preprocessing.preprocess_danbooru_post_ratings as prep


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("", False),
        ("a", True),
        ("ab", True),
        ("long_hair", True),
        ("123", False),
        ("a1", False),
        ("_", False),
        ("__", False),
        ("a_", True),
        ("_a", True),
    ],
)
def test_is_tag_valid(tag: str, expected: bool) -> None:
    assert prep.is_tag_valid(tag) is expected


def test_tag_string_from_general_tags_dedup_sort_and_underscores() -> None:
    tags = [
        {"category": "0", "name": "a"},
        {"category": "0", "name": "a_b"},
        {"category": "1", "name": "ignored"},
        {"category": "0", "name": "zebra"},
    ]
    assert prep.tag_string_from_general_tags(tags) == "a b zebra"


def test_tag_string_from_general_tags_skips_invalid() -> None:
    tags = [
        {"category": "0", "name": "valid_tag"},
        {"category": "0", "name": "bad-dash"},
    ]
    assert prep.tag_string_from_general_tags(tags) == "tag valid"


def test_stream_jsonl_batches(tmp_path: Path) -> None:
    path = tmp_path / "lines.json"
    rows = [{"id": i} for i in range(5)]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    batches = list(prep.stream_jsonl(path, batch_size=2))
    assert batches == [[rows[0], rows[1]], [rows[2], rows[3]], [rows[4]]]


def test_transform_post_rating_and_tags() -> None:
    post = {
        "rating": "E",
        "tags": [
            {"category": "0", "name": "solo"},
            {"category": "0", "name": "long_hair"},
        ],
    }
    out = prep.transform_post(post)
    assert out["rating"] == "nsfw"
    assert out["tag_string"] == "hair long solo"


def test_transform_post_unknown_rating_defaults_sfw() -> None:
    post = {"rating": "x", "tags": []}
    assert prep.transform_post(post)["rating"] == "sfw"


def test_lemmatize_post_batch_sorts_unique_lemmas() -> None:
    posts = [{"tag_string": "dog cat dog", "rating": "sfw"}]
    out = prep.lemmatize_post_batch(posts)
    assert out[0]["tag_string"] == "cat dog"
