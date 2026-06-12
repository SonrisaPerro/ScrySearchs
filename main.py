import asyncio
import os
import gdown
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Deque, List

import faiss
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager
from io import BytesIO

ROOT_DIR = Path(__file__).resolve().parent
INDEX_PATH = ROOT_DIR / "scryfall_index.faiss"
MAPPING_PATH = ROOT_DIR / "id_mapping.json"
MODEL_DIR = ROOT_DIR / "clip-model"

ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv(
    "ALLOWED_ORIGINS",
    "http://127.0.0.1:8000,http://localhost:8000"
).split(",") if origin.strip()]
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

index: faiss.Index = None  # type: ignore[assignment]
model: SentenceTransformer = None  # type: ignore[assignment]
id_mapping: List[Dict[str, Any]] = []

rate_limit_store: Dict[str, Deque[float]] = defaultdict(deque)
rate_limit_lock = asyncio.Lock()


def load_index(path: Path) -> faiss.Index:
    if not path.exists():
        raise RuntimeError(f"Missing FAISS index at {path}")

    index_obj = faiss.read_index(str(path))
    if index_obj.ntotal == 0:
        raise RuntimeError("FAISS index is empty.")

    return index_obj


@asynccontextmanager
async def lifespan(app: FastAPI):
    global index, model, id_mapping

    print("Loading database and model into memory...")

    if not MAPPING_PATH.exists():
        raise RuntimeError(f"Missing ID mapping at {MAPPING_PATH}")

    with MAPPING_PATH.open("r", encoding="utf-8") as f:
        id_mapping = json.load(f)

    # --- CLOUD DOWNLOAD WORKAROUND ---
    DRIVE_FILE_ID = "1LCWxaFKxKPLx4ss2uDBSuHf0oQnVIe7d"
    if not INDEX_PATH.exists():
        print("Downloading FAISS index from Google Drive. Please wait...")
        download_url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(download_url, str(INDEX_PATH), quiet=False)
    # ---------------------------------

    index = load_index(INDEX_PATH)
    model = SentenceTransformer('clip-ViT-B-32')

    print(f"Backend ready! Loaded {index.ntotal} cards into memory.")
    yield
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def simple_rate_limiter(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
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


@app.post("/search")
async def search_image(file: UploadFile = File(...)) -> Dict[str, List[Dict[str, Any]]]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image file.")

    embedding = model.encode([img], convert_to_numpy=True)
    if embedding.ndim == 2:
        embedding = embedding[0]

    embedding_matrix = np.asarray([embedding], dtype="float32")
    faiss.normalize_L2(embedding_matrix)

    k = min(30, int(index.ntotal))
    if k <= 0:
        raise HTTPException(status_code=500, detail="Search index contains no vectors.")

    distances, indices = index.search(embedding_matrix, k)

    results: List[Dict[str, Any]] = []
    for score, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(id_mapping):
            continue

        card_meta = id_mapping[int(idx)]
        results.append(
            {
                "scryfall_id": card_meta["scryfall_id"],
                "name": card_meta["name"],
                "similarity_score": round(float(score), 4),
                "api_link": f"https://api.scryfall.com/cards/{card_meta['scryfall_id']}"
            }
        )

    return {"matches": results}


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def serve_frontend() -> FileResponse:
    return FileResponse(ROOT_DIR / "index.html")