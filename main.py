import asyncio
import os
import json
import threading
import warnings
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Deque, List, Optional

import faiss
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager
from io import BytesIO

from assets import INDEX_PATH, MAPPING_PATH, MODEL_NAME, ROOT_DIR, ensure_assets

ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv(
    "ALLOWED_ORIGINS",
    "http://127.0.0.1:8000,http://localhost:8000"
).split(",") if origin.strip()]
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_QUERY_CHARS = 200
RESULT_COUNT = 30

# A small file can declare a huge canvas (decompression bomb). Reject past 120 MP, which
# still admits 108 MP phone photos; PIL's default only errors at twice its warning limit.
Image.MAX_IMAGE_PIXELS = 120_000_000
warnings.simplefilter("error", Image.DecompressionBombWarning)
# Decoding and CLIP are memory/CPU heavy; cap concurrent searches so bursts queue instead of OOMing.
search_slots = threading.BoundedSemaphore(4)

index: faiss.Index = None  # type: ignore[assignment]
model: SentenceTransformer = None  # type: ignore[assignment]
id_mapping: List[Dict[str, Any]] = []
illustration_rows: Dict[str, int] = {}

rate_limit_store: Dict[str, Deque[float]] = defaultdict(deque)
rate_limit_lock = asyncio.Lock()


# Real cards are 63x88mm (0.716); the range allows for screenshot margins.
CARD_ASPECT_RANGE = (0.62, 0.80)
# Art box of a standard (1993-2015+) frame, as fractions of the full card.
ART_BOX = (0.08, 0.11, 0.92, 0.555)


def crop_art_box(img: Image.Image) -> Image.Image:
    w, h = img.size
    left, top, right, bottom = ART_BOX
    return img.crop((int(w * left), int(h * top), int(w * right), int(h * bottom)))


def load_index(path: Path) -> faiss.Index:
    if not path.exists():
        raise RuntimeError(f"Missing FAISS index at {path}")

    index_obj = faiss.read_index(str(path))
    if index_obj.ntotal == 0:
        raise RuntimeError("FAISS index is empty.")

    return index_obj


@asynccontextmanager
async def lifespan(app: FastAPI):
    global index, model, id_mapping, illustration_rows

    print("Loading database and model into memory...")

    # A no-op in the Docker image, which bakes these in at build time.
    ensure_assets()

    with MAPPING_PATH.open("r", encoding="utf-8") as f:
        id_mapping = json.load(f)

    index = load_index(INDEX_PATH)
    if index.ntotal != len(id_mapping):
        raise RuntimeError(f"Index has {index.ntotal} vectors but mapping has {len(id_mapping)} entries.")
    illustration_rows = {entry["illustration_id"]: row for row, entry in enumerate(id_mapping)
                         if entry.get("illustration_id")}
    model = SentenceTransformer(MODEL_NAME)

    print(f"Backend ready! Loaded {index.ntotal} cards into memory.")
    yield
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def simple_rate_limiter(request: Request, call_next):
    # Behind Railway's proxy request.client is the proxy; Railway puts the visitor in X-Real-IP.
    client_ip = request.headers.get("x-real-ip") or (request.client.host if request.client else "unknown")
    now = asyncio.get_event_loop().time()
    window_start = now - RATE_LIMIT_WINDOW

    async with rate_limit_lock:
        request_times = rate_limit_store[client_ip]
        while request_times and request_times[0] <= window_start:
            request_times.popleft()

        if len(request_times) >= RATE_LIMIT_REQUESTS:
            return JSONResponse(
                {"detail": "Too many requests. Please try again later."},
                status_code=429,
            )

        request_times.append(now)

    return await call_next(request)


# Plain def: FastAPI runs it in a worker thread, so CLIP inference doesn't block other requests.
@app.post("/search")
def search_image(
    file: UploadFile = File(...),
    leniency: float = Form(0.25)
) -> Dict[str, List[Dict[str, Any]]]:
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large (max 20 MB).")

    with search_slots:
        return {"matches": _search(contents, leniency)}


def _search(contents: bytes, leniency: float) -> List[Dict[str, Any]]:
    try:
        img = Image.open(BytesIO(contents))
        img.draft("RGB", (1024, 1024))  # JPEG decodes at reduced scale; no-op for other formats
        img = img.convert("RGB")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(status_code=413, detail="Image dimensions are too large (max 120 megapixels).")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image file.")

    # The index holds art crops only, so a full-card upload (screenshot/scan) also gets
    # searched as its art box; each card keeps its better score.
    queries = [img]
    if CARD_ASPECT_RANGE[0] <= img.width / img.height <= CARD_ASPECT_RANGE[1]:
        queries.append(crop_art_box(img))

    return _nearest(model.encode(queries, convert_to_numpy=True), leniency)


def _nearest(vectors: Any, leniency: Optional[float], exclude_row: Optional[int] = None) -> List[Dict[str, Any]]:
    """Top matches for one or more query vectors; each card keeps its best score."""
    matrix = np.asarray(vectors, dtype="float32").reshape(-1, index.d)
    faiss.normalize_L2(matrix)
    distances, rows = index.search(matrix, RESULT_COUNT + 1)

    best: Dict[int, float] = {}
    for score, row in zip(distances.ravel(), rows.ravel()):
        if 0 <= row < len(id_mapping) and row != exclude_row:
            best[int(row)] = max(best.get(int(row), -1.0), float(score))

    results: List[Dict[str, Any]] = []
    for row, score in sorted(best.items(), key=lambda item: item[1], reverse=True)[:RESULT_COUNT]:
        # The Leniency Filter: IndexFlatIP scores are cosine similarity (higher = closer),
        # so compare cosine distance (0..2) against the slider.
        if leniency is not None and 1.0 - score > leniency:
            continue

        card_meta = id_mapping[row]
        results.append(
            {
                "scryfall_id": card_meta["scryfall_id"],
                "name": card_meta["name"],
                "illustration_id": card_meta.get("illustration_id"),
                "similarity_score": round(score, 4),
                "api_link": f"https://api.scryfall.com/cards/{card_meta['scryfall_id']}"
            }
        )

    return results


# Text and image vectors share CLIP's space, but text scores sit on a compressed scale
# (~0.23-0.34 for the whole top 30), so the image leniency slider doesn't apply here.
@app.post("/search/text")
def search_text(query: str = Form("")) -> Dict[str, List[Dict[str, Any]]]:
    query = query.strip()
    if not query or len(query) > MAX_QUERY_CHARS:
        raise HTTPException(status_code=400, detail=f"Describe the art in 1 to {MAX_QUERY_CHARS} characters.")
    with search_slots:
        return {"matches": _nearest(model.encode([query], convert_to_numpy=True), leniency=None)}


# Uses the stored vector, so there is no image to download or embed.
@app.get("/similar")
def similar_art(illustration_id: str = Query(...), leniency: float = Query(0.25)) -> Dict[str, Any]:
    row = illustration_rows.get(illustration_id)
    if row is None:
        raise HTTPException(status_code=404, detail="That artwork isn't in the index.")
    source = id_mapping[row]
    return {
        "source": {"scryfall_id": source["scryfall_id"], "name": source["name"], "illustration_id": illustration_id},
        "matches": _nearest(index.reconstruct(row), leniency, exclude_row=row),
    }


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def serve_frontend() -> FileResponse:
    return FileResponse(ROOT_DIR / "index.html")