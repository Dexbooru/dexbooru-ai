import json
import logging
import re
import sys
from collections import defaultdict
from collections.abc import Generator
from pathlib import Path

import pandas as pd
import spacy

logger = logging.getLogger(__name__)

_PREPROCESSING_DIR = Path(__file__).resolve().parent
_TRAINING_DIR = _PREPROCESSING_DIR.parent
RAW_DANBOORU_POSTS_DIR = _TRAINING_DIR / "raw_data" / "danbooru_posts" / "original"
OUTPUT_DIR = _TRAINING_DIR / "raw_data" / "danbooru_posts" / "final"

NSFW_RATINGS_MAP = {
    "s": "sfw",
    "q": "likely_nsfw",
    "e": "nsfw",
}

MAXIMUM_POSTS_PER_RATING_CATEGORY = 60_000
MAX_BATCH_FOR_CSV = 10_000
MAXIMUM_DATASET_SIZE = MAXIMUM_POSTS_PER_RATING_CATEGORY * len(NSFW_RATINGS_MAP)
STREAM_BATCH_SIZE = 1000
PROGRESS_LOG_INTERVAL = 10_000

_TAG_NAME_RE = re.compile(r"^[A-Za-z_]+$")

nlp = spacy.load("en_core_web_md", disable=["ner", "parser"])


def is_tag_valid(tag: str) -> bool:
    if not tag or not _TAG_NAME_RE.fullmatch(tag):
        return False
    return any(c.isalpha() for c in tag)


def tag_string_from_general_tags(tags: list[dict]) -> str:
    words: set[str] = set()
    for tag in tags:
        if tag.get("category") != "0":
            continue
        name = tag.get("name") or ""
        if not is_tag_valid(name):
            continue
        for part in name.replace("_", " ").split():
            words.add(part)
    return " ".join(sorted(words))


def stream_jsonl(file_path: str | Path, batch_size: int = 1000) -> Generator[dict, None, None]:
    current_batch: list[dict] = []

    with open(file_path) as f:
        for line in f:
            if batch_size == 1:
                yield json.loads(line)

            current_batch.append(json.loads(line))
            if len(current_batch) >= batch_size:
                yield current_batch
                current_batch = []

        if current_batch:
            yield current_batch


def transform_post(post: dict) -> dict:
    rating = post.get("rating")
    tags = post.get("tags", [])

    new_rating = NSFW_RATINGS_MAP.get(rating.lower(), "sfw")
    tag_string = tag_string_from_general_tags(tags)

    return {
        "rating": new_rating,
        "tag_string": tag_string,
    }


def lemmatize_post_batch(posts: list[dict]) -> list[dict]:
    batch_tag_strings = [post.get("tag_string") for post in posts]
    lemmatized_docs = nlp.pipe(batch_tag_strings, n_process=-1, batch_size=len(posts))

    for post, doc in zip(posts, lemmatized_docs):
        lemmas = sorted({token.lemma_ for token in doc if token.lemma_.strip()})
        post["tag_string"] = " ".join(lemmas)

    return posts


def main() -> None:
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        post_json_files = sorted(
            p.name
            for p in RAW_DANBOORU_POSTS_DIR.iterdir()
            if p.is_file() and p.suffix == ".json"
        )
        logger.info("Found %d JSON post files under %s", len(post_json_files), RAW_DANBOORU_POSTS_DIR)

        counts: defaultdict[str, int] = defaultdict(int)
        buffer_dataset: list[dict] = []
        csv_index = 0
        last_logged_total = 0

        for filename in post_json_files:
            if sum(counts.values()) >= MAXIMUM_DATASET_SIZE:
                break

            batch_stream = stream_jsonl(RAW_DANBOORU_POSTS_DIR / filename, STREAM_BATCH_SIZE)

            for batch in batch_stream:
                transformed_posts = [transform_post(post) for post in batch]

                needed_posts = []
                for post in transformed_posts:
                    rating = post["rating"]
                    if counts[rating] < MAXIMUM_POSTS_PER_RATING_CATEGORY:
                        needed_posts.append(post)
                        counts[rating] += 1

                if not needed_posts:
                    continue

                transformed_posts_with_lemmas = lemmatize_post_batch(needed_posts)
                buffer_dataset.extend(transformed_posts_with_lemmas)

                total = sum(counts.values())
                stats = " | ".join(f"{k.upper()}: {v}" for k, v in sorted(counts.items()))
                if total - last_logged_total >= PROGRESS_LOG_INTERVAL or total >= MAXIMUM_DATASET_SIZE:
                    logger.info("Progress %d / %d posts | %s", total, MAXIMUM_DATASET_SIZE, stats)
                    last_logged_total = total

                if len(buffer_dataset) >= MAX_BATCH_FOR_CSV:
                    out_path = OUTPUT_DIR / f"tag_string_rating-{csv_index}.csv"
                    df = pd.DataFrame(buffer_dataset, columns=["tag_string", "rating"])
                    df.to_csv(out_path, index=False)
                    logger.info("Wrote %s (%d rows)", out_path, len(df))
                    buffer_dataset = []
                    csv_index += 1

                if sum(counts.values()) >= MAXIMUM_DATASET_SIZE:
                    break

        if buffer_dataset:
            out_path = OUTPUT_DIR / f"tag_string_rating-{csv_index}.csv"
            df = pd.DataFrame(buffer_dataset, columns=["tag_string", "rating"])
            df.to_csv(out_path, index=False)
            logger.info("Wrote %s (%d rows)", out_path, len(df))

        logger.info("Collection summary:")
        for rating, count in sorted(counts.items()):
            logger.info("  %s: %d posts", rating.upper(), count)
    except Exception as e:
        logger.error("Preprocessing failed: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        main()
    except Exception:
        sys.exit(1)
