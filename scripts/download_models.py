from __future__ import annotations

import argparse
from pathlib import Path
import sys
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

MODEL_URLS = {
    "candy": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/candy.t7",
    "mosaic": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/mosaic.t7",
    "udnie": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/udnie.t7",
    "the_scream": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/the_scream.t7",
    "la_muse": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/eccv16/la_muse.t7",
    "starry_night": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/eccv16/starry_night.t7",
    "the_wave": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/eccv16/the_wave.t7",
    "composition_vii": "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/eccv16/composition_vii.t7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download pretrained OpenCV-compatible style models.")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_URLS),
        default="candy",
        help="Model name to download.",
    )
    parser.add_argument("--all", action="store_true", help="Download every configured model.")
    parser.add_argument("--force", action="store_true", help="Download again even if the file exists.")
    return parser.parse_args()


def progress_hook(block_number: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = block_number * block_size
    percent = min(100, downloaded * 100 / total_size)
    sys.stdout.write(f"\r  {percent:6.2f}%")
    sys.stdout.flush()


def download_model(name: str, url: str, force: bool = False) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    destination = MODELS_DIR / f"{name}.t7"

    if destination.exists() and not force:
        print(f"Skipping {name}: {destination} already exists")
        return destination

    print(f"Downloading {name} from {url}")
    urllib.request.urlretrieve(url, destination, reporthook=progress_hook)
    print(f"\nSaved to {destination}")
    return destination


def main() -> None:
    args = parse_args()
    selected = MODEL_URLS if args.all else {args.model: MODEL_URLS[args.model]}

    for name, url in selected.items():
        download_model(name, url, force=args.force)

    print("Model download complete.")


if __name__ == "__main__":
    main()
