import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = ROOT_DIR / "images"
INDEX_PATH = ROOT_DIR / "scryfall_index.faiss"
MODEL_DIR = ROOT_DIR / "clip-model"
MAPPING_PATH = ROOT_DIR / "id_mapping.json"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

USER_AGENT = "MTGReverseImageSearch/1.0 (contact@example.com)"
BULK_META_URL = "https://api.scryfall.com/bulk-data"


def load_clip_model(model_path: Path) -> SentenceTransformer:
    if not model_path.exists():
        raise FileNotFoundError(f"CLIP model directory not found: {model_path}")

    logger.info("Loading CLIP model from %s", model_path)
    return SentenceTransformer(str(model_path))


def fetch_bulk_cards(session: requests.Session) -> List[Dict[str, any]]:
    logger.info("Fetching Scryfall bulk data metadata")
    response = session.get(BULK_META_URL)
    response.raise_for_status()

    bulk_types = response.json().get("data", [])
    default_cards = next((item for item in bulk_types if item.get("type") == "default_cards"), None)
    if default_cards is None:
        raise RuntimeError("Could not find default_cards bulk data in Scryfall response.")

    download_url = default_cards["download_uri"]
    logger.info("Downloading card catalog from %s", download_url)

    response = session.get(download_url)
    response.raise_for_status()
    return response.json()


def gather_unique_artwork(cards_data: List[Dict[str, any]]) -> List[Dict[str, str]]:
    unique_art_cards: Dict[str, Dict[str, str]] = {}
    for card in cards_data:
        if "image_uris" in card and card["image_uris"].get("art_crop"):
            art_id = card.get("illustration_id")
            if art_id and art_id not in unique_art_cards:
                unique_art_cards[art_id] = {
                    "scryfall_id": card["id"],
                    "name": card["name"],
                    "art_url": card["image_uris"]["art_crop"],
                }

        elif "card_faces" in card:
            for face in card["card_faces"]:
                if face.get("image_uris") and face["image_uris"].get("art_crop"):
                    art_id = face.get("illustration_id")
                    if art_id and art_id not in unique_art_cards:
                        unique_art_cards[art_id] = {
                            "scryfall_id": card["id"],
                            "name": f"{card['name']} ({face['name']})",
                            "art_url": face["image_uris"]["art_crop"],
                        }

    return list(unique_art_cards.values())


def download_image(session: requests.Session, url: str) -> Optional[Image.Image]:
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as exc:
        logger.debug("Failed to download image %s: %s", url, exc)
        return None


def build_embeddings(
    session: requests.Session,
    model: SentenceTransformer,
    cards: List[Dict[str, str]],
    limit: Optional[int] = None,
) -> (np.ndarray, List[Dict[str, str]]):
    embeddings: List[np.ndarray] = []
    id_mapping: List[Dict[str, str]] = []

    cards_to_process = cards[:limit] if limit is not None else cards

    logger.info("Processing %d artworks", len(cards_to_process))
    for card in tqdm(cards_to_process, desc="Embedding images", unit="cards"):
        image = download_image(session, card["art_url"])
        if image is None:
            continue

        try:
            embedding = model.encode([image], convert_to_numpy=True)
            embedding = np.asarray(embedding, dtype="float32")
            if embedding.ndim == 2:
                embedding = embedding[0]

            embeddings.append(embedding)
            id_mapping.append({"scryfall_id": card["scryfall_id"], "name": card["name"]})
        except Exception as exc:
            logger.debug("Skipping artwork %s due to embed error: %s", card["art_url"], exc)
            continue

    if not embeddings:
        raise RuntimeError("No embeddings were generated. Check download and model configuration.")

    embedding_matrix = np.vstack(embeddings)
    faiss.normalize_L2(embedding_matrix)
    return embedding_matrix, id_mapping


def build_index(embeddings_matrix: np.ndarray) -> faiss.IndexFlatIP:
    dimension = embeddings_matrix.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_matrix)
    return index


def save_artifacts(index: faiss.IndexFlatIP, id_mapping: List[Dict[str, str]]) -> None:
    faiss.write_index(index, str(INDEX_PATH))
    with MAPPING_PATH.open("w", encoding="utf-8") as f:
        json.dump(id_mapping, f, indent=2)


def main(limit: Optional[int] = None) -> None:
    model = load_clip_model(MODEL_DIR)
    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT})
        cards_data = fetch_bulk_cards(session)

        unique_art_cards = gather_unique_artwork(cards_data)
        logger.info("Found %d unique artworks to index.", len(unique_art_cards))

        embeddings_matrix, id_mapping = build_embeddings(session, model, unique_art_cards, limit=limit)
        logger.info("Compiled %d embeddings.", len(id_mapping))

        faiss_index = build_index(embeddings_matrix)
        save_artifacts(faiss_index, id_mapping)

        logger.info("Success! Vector database saved to %s. Indexed %d cards.", INDEX_PATH, faiss_index.ntotal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a FAISS image search index for Scryfall cards.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on the number of artworks to process.")
    args = parser.parse_args()
    main(limit=args.limit)
