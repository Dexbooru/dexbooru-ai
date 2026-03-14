import os

VERSION_FILE_PATH = "./version"


def get_version() -> str:
    if not os.path.exists(VERSION_FILE_PATH):
        return "0.0.0"

    with open(VERSION_FILE_PATH) as f:
        return f.read().strip().split(".")[0]
