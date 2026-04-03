#!/usr/bin/env python3
"""Download TMDB 5000 movie metadata from Kaggle into inputs/tmdb-movie-metadata/."""

import shutil
import sys
from pathlib import Path

import kagglehub

DATASET = "tmdb/tmdb-movie-metadata"
DEST_DIRNAME = "tmdb-movie-metadata"


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    dest = root / "inputs" / DEST_DIRNAME
    dest.parent.mkdir(parents=True, exist_ok=True)

    path = kagglehub.dataset_download(DATASET)
    src = Path(path)
    print("Path to dataset files (kagglehub cache):", path)

    if not src.exists():
        print(f"Download path not found: {src}", file=sys.stderr)
        return 1

    if dest.exists():
        shutil.rmtree(dest)
    if src.is_dir():
        shutil.copytree(src, dest)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest / src.name)

    print("Copied to:", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
