"""Where the search assets live, shared by base.py (builds them) and main.py (serves them).

The FAISS index and its id mapping are published together as a GitHub Release;
index_version.txt names the release the app should use.
"""
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
INDEX_PATH = ROOT_DIR / "scryfall_index.faiss"
MAPPING_PATH = ROOT_DIR / "id_mapping.json"
VERSION_PATH = ROOT_DIR / "index_version.txt"

# base.py and main.py must embed with the same model, or every search silently breaks.
MODEL_NAME = "clip-ViT-B-32"

RELEASE_URL = "https://github.com/SonrisaPerro/ScrySearchs/releases/download/{tag}/{name}"

# Filter metadata stored per artwork in the mapping as small bitmasks (base.py writes, main.py reads).
COLOR_BITS = {"W": 1, "U": 2, "B": 4, "R": 8, "G": 16}
TYPE_BITS = {"creature": 1, "land": 2, "instant": 4, "sorcery": 4, "artifact": 8, "enchantment": 16, "planeswalker": 32, "battle": 64}
FORMATS = ["commander", "standard", "pioneer", "modern", "legacy", "vintage", "pauper"]
ODD_LAYOUTS = {"token", "double_faced_token", "emblem", "planar", "vanguard", "scheme"}


def ensure_assets() -> None:
    """Download the index and mapping from the pinned release if they aren't on disk."""
    for path in (INDEX_PATH, MAPPING_PATH):
        if path.exists():
            continue
        tag = VERSION_PATH.read_text(encoding="utf-8").strip()
        print(f"Downloading {path.name} from release {tag}...")
        partial = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(RELEASE_URL.format(tag=tag, name=path.name), partial)
        partial.replace(path)
