import argparse
import gzip
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from collections import Counter
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import requests
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from assets import (COLOR_BITS, FORMATS, INDEX_PATH, MAPPING_PATH, MODEL_NAME, ODD_LAYOUTS, TYPE_BITS,
                    ensure_assets)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

USER_AGENT = "ScrySearchs/1.0 (https://github.com/SonrisaPerro/ScrySearchs)"
BULK_META_URL = "https://api.scryfall.com/bulk-data"
BATCH_SIZE = 64
DOWNLOAD_WORKERS = 8  # Scryfall's image CDN isn't rate limited, only api.scryfall.com is
# Scryfall serves a stand-in image until a card's real scan exists; embedding it would pollute the
# index, so those artworks wait for a later refresh.
PLACEHOLDER_STATUSES = {"missing", "placeholder"}
MAPPING_FIELDS = ("scryfall_id", "name", "illustration_id", "ci", "t", "f", "odd")


def filter_metadata(card: Dict[str, Any], type_line: str) -> Dict[str, int]:
    """Colour identity, card types, legal formats and token-ness as the bitmasks main.py filters on."""
    legal = card.get("legalities", {})
    type_words = set(re.findall(r"[a-z]+", type_line.lower()))
    return {
        "ci": sum(COLOR_BITS[c] for c in card.get("color_identity", []) if c in COLOR_BITS),
        "t": sum({bit for word, bit in TYPE_BITS.items() if word in type_words}),
        "f": sum(1 << i for i, fmt in enumerate(FORMATS) if legal.get(fmt) in ("legal", "restricted")),
        "odd": int(card.get("layout") in ODD_LAYOUTS),
    }


def fetch_artworks(session: requests.Session) -> List[Dict[str, str]]:
    """One entry per illustration, from Scryfall's unique_artwork bulk file."""
    bulk_types = session.get(BULK_META_URL).json()["data"]
    url = next(b["jsonl_download_uri"] for b in bulk_types if b["type"] == "unique_artwork")
    logger.info("Downloading %s", url)
    response = session.get(url)
    response.raise_for_status()

    artworks: Dict[str, Dict[str, str]] = {}
    for line in gzip.decompress(response.content).splitlines():
        card = json.loads(line)
        # Art Series cards are collectible art prints, not playable cards, and their backs all share
        # one generic logo image.
        if card.get("layout") == "art_series" or card.get("image_status") in PLACEHOLDER_STATUSES:
            continue
        # Split/flip/adventure cards share one image; double-faced cards have one per face.
        faces = [card] if "image_uris" in card else card.get("card_faces", [])
        for face in faces:
            art_id = face.get("illustration_id")
            art_url = (face.get("image_uris") or {}).get("art_crop")
            if art_id and art_url and art_id not in artworks:
                name = card["name"] if face is card else f"{card['name']} ({face['name']})"
                type_line = face.get("type_line") or card.get("type_line", "")
                artworks[art_id] = {"scryfall_id": card["id"], "name": name, "illustration_id": art_id,
                                    "art_url": art_url, **filter_metadata(card, type_line)}

    # Stable order, so an unchanged catalog produces a byte-identical mapping.
    return sorted(artworks.values(), key=lambda a: (a["name"], a["illustration_id"]))


def load_previous_vectors() -> Dict[str, np.ndarray]:
    if not (INDEX_PATH.exists() and MAPPING_PATH.exists()):
        return {}
    index = faiss.read_index(str(INDEX_PATH))
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    vectors = index.reconstruct_n(0, index.ntotal)
    # Different artworks never embed identically; a vector shared by several is a placeholder image
    # that slipped through, so drop it and let this run embed the real art.
    counts = Counter(v.tobytes() for v in vectors)
    return {entry["illustration_id"]: vectors[i] for i, entry in enumerate(mapping)
            if entry.get("illustration_id") and counts[vectors[i].tobytes()] <= 2}


def download_image(session: requests.Session, url: str) -> Optional[Image.Image]:
    try:
        response = session.get(url, timeout=20)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as exc:
        logger.warning("Failed to download %s: %s", url, exc)
        return None


def embed_artworks(session: requests.Session, model: SentenceTransformer,
                   artworks: List[Dict[str, str]]) -> Dict[str, np.ndarray]:
    vectors: Dict[str, np.ndarray] = {}
    with ThreadPoolExecutor(DOWNLOAD_WORKERS) as pool, tqdm(total=len(artworks), desc="Embedding", unit="art") as bar:
        # One batch in memory at a time; pool.map over everything would buffer every image.
        for start in range(0, len(artworks), BATCH_SIZE):
            batch = artworks[start:start + BATCH_SIZE]
            images = list(pool.map(lambda a: download_image(session, a["art_url"]), batch))
            # Failed downloads are skipped this run and retried on the next refresh.
            ok = [(a["illustration_id"], img) for a, img in zip(batch, images) if img is not None]
            if ok:
                embedded = model.encode([img for _, img in ok], batch_size=BATCH_SIZE, convert_to_numpy=True,
                                        show_progress_bar=False)
                embedded = np.asarray(embedded, dtype="float32")
                faiss.normalize_L2(embedded)
                vectors.update(zip([art_id for art_id, _ in ok], embedded))
            bar.update(len(batch))
    return vectors


def main(full: bool = False, limit: Optional[int] = None) -> None:
    if not full:
        ensure_assets()
    previous = {} if full else load_previous_vectors()

    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})
        artworks = fetch_artworks(session)[:limit]
        new = [a for a in artworks if a["illustration_id"] not in previous]
        logger.info("%d artworks on Scryfall, %d already indexed, %d to embed.",
                    len(artworks), len(artworks) - len(new), len(new))

        if new:
            model = SentenceTransformer(MODEL_NAME)
            previous.update(embed_artworks(session, model, new))

    kept = [a for a in artworks if a["illustration_id"] in previous]
    matrix = np.vstack([previous[a["illustration_id"]] for a in kept]).astype("float32")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    faiss.write_index(index, str(INDEX_PATH))
    mapping = [{k: a[k] for k in MAPPING_FIELDS} for a in kept]
    MAPPING_PATH.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved %d artworks to %s", index.ntotal, INDEX_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build or refresh the FAISS art index from Scryfall.")
    parser.add_argument("--full", action="store_true", help="Re-embed everything instead of reusing the current index.")
    parser.add_argument("--limit", type=int, default=None, help="Only index the first N artworks (for development).")
    args = parser.parse_args()
    main(full=args.full, limit=args.limit)
