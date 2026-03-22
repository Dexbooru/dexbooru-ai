import logging
import os
import random
import shutil
import tempfile
import time
import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm

# --- Configuration ---
DATASET_NAME = "mylesoneill/tagged-anime-illustrations"
BASE_PATH = "danbooru-metadata/danbooru-metadata/"
LOCAL_TARGET_FOLDER = "../raw_data/danbooru_posts"
MAX_RETRIES = 5
COOLDOWN_SECONDS = 1.5  # Forced wait between successful downloads

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def download_file_with_retry(api, full_path, tmp_dir, file_name):
    """Synchronous download with exponential backoff for 429s."""
    for attempt in range(MAX_RETRIES):
        try:
            # quiet=True stops Kaggle's internal progress bar from breaking tqdm
            api.dataset_download_file(DATASET_NAME, full_path, path=tmp_dir, quiet=True)
            return True
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg:
                # Exponential backoff: 5s, 10s, 20s... with jitter
                wait_time = (5 * (2 ** attempt)) + (random.random() * 5)
                logger.warning(f"Rate limited (429) on {file_name}. Waiting {wait_time:.2f}s...")
                time.sleep(wait_time)
            elif "404" in err_msg:
                return False  # Chunk doesn't exist
            else:
                logger.error(f"Unexpected error on {file_name}: {e}")
                raise e
    return False

def main() -> None:
    api = KaggleApi()
    api.authenticate()

    os.makedirs(LOCAL_TARGET_FOLDER, exist_ok=True)

    years = range(2014, 2019)
    chunks = range(0, 25)

    # Flatten the task list for the progress bar
    tasks = [(y, c) for y in years for c in chunks]

    logger.info(f"Starting synchronous sync to {LOCAL_TARGET_FOLDER}")

    for year, chunk in tqdm(tasks, desc="Syncing Metadata", unit="file"):
        file_name = f"{year}{chunk:02d}.json"
        dest_name = f"{year}_{chunk:02d}_posts.json"
        dest_path = os.path.join(LOCAL_TARGET_FOLDER, dest_name)

        # Skip if already downloaded (Idempotency)
        if os.path.exists(dest_path):
            continue

        full_kaggle_path = f"{BASE_PATH}{file_name}"

        with tempfile.TemporaryDirectory() as tmp_dir:
            if download_file_with_retry(api, full_kaggle_path, tmp_dir, file_name):
                zip_path = os.path.join(tmp_dir, f"{file_name}.zip")

                if os.path.exists(zip_path):
                    with zipfile.ZipFile(zip_path, 'r') as z:
                        z.extractall(tmp_dir)

                    src_json = os.path.join(tmp_dir, file_name)
                    if os.path.exists(src_json):
                        shutil.move(src_json, dest_path)
                        # Mandatory cooldown to prevent burst-triggering the 429
                        time.sleep(COOLDOWN_SECONDS + random.random())
                else:
                    # Some files might not come down as zips if they are small
                    src_json = os.path.join(tmp_dir, file_name)
                    if os.path.exists(src_json):
                        shutil.move(src_json, dest_path)
                        time.sleep(COOLDOWN_SECONDS)

    logger.info("Sync complete. All files accounted for.")

if __name__ == "__main__":
    main()
