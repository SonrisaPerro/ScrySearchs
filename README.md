# scryfallsearch
# Scryfall Search

A local Magic: The Gathering reverse image search web app built with FastAPI, FAISS, and CLIP embeddings.

## Overview

This project indexes Scryfall artwork and enables image-based card matching.

- `base.py`: Builds a FAISS similarity index from Scryfall artwork using a local CLIP model.
- `main.py`: FastAPI backend that serves a search endpoint and the frontend.
- `launch.py`: Helper script to start the backend, verify required assets, and open the browser.
- `index.html`: Client UI for uploading images, filtering results, and viewing card matches.
- `Dockerfile`: Containerized runtime for the app.
- `Procfile`: Heroku-compatible startup command.

## Features

- Reverse image search using image embeddings
- Local vector search with FAISS
- FastAPI backend with `/search` and `/health`
- Browser frontend with drag-and-drop, clipboard paste, and file upload
- Result filtering for nonstandard card layouts and Commander legality
- Automatic card metadata lookup from Scryfall API
- Optional affiliate redirect support for TCGplayer links

## Requirements

- Python 3.12 (or compatible Python 3.x)
- `pip`
- `faiss-cpu`
- `fastapi`
- `uvicorn[standard]`
- `sentence-transformers`
- `numpy`
- `Pillow`
- `requests`
- `tqdm`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Required Assets

The app expects these files/folders in the repository root:

- `clip-model/` — local SentenceTransformers CLIP model directory
- `scryfall_index.faiss` — FAISS vector index file
- `id_mapping.json` — card metadata mapping for search results
- `images/` — optional artwork cache directory created by `base.py`

If assets are missing, `launch.py` can optionally build them using `base.py`.

## Building the Search Index

Use `base.py` to download Scryfall bulk data, fetch artwork, embed images, and save the vector index.

```bash
python base.py
```

To build a smaller sample index for development:

```bash
python base.py --limit 1000
```

The builder will:

1. Download Scryfall bulk card data.
2. Gather unique artwork URLs.
3. Download and embed each artwork with the CLIP model.
4. Normalize vectors and save a FAISS index.
5. Save `id_mapping.json` with card IDs and names.

## Running Locally

### Option 1: Start with `launch.py`

This script checks for required assets, starts the backend, and opens the browser.

```bash
python launch.py
```

### Option 2: Run `uvicorn` directly

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000` in your browser.

## API Endpoints

### `POST /search`

Accepts a multipart form upload with field `file`.

Request example:

```bash
curl -X POST "http://127.0.0.1:8000/search" \
  -F "file=@/path/to/image.jpg"
```

Response format:

```json
{
  "matches": [
    {
      "scryfall_id": "...",
      "name": "...",
      "similarity_score": 0.1234,
      "api_link": "https://api.scryfall.com/cards/..."
    }
  ]
}
```

### `GET /health`

Returns a simple health check JSON:

```json
{ "status": "ok" }
```

### `GET /`

Serves the frontend from `index.html`.

## Frontend Usage

Open the web app in your browser and:

- Drag and drop an image onto the search zone
- Paste an image from the clipboard
- Choose a file using the upload button
- Adjust filters and maximum results
- Click a result card to zoom and flip double-faced cards

The frontend fetches additional Scryfall metadata for each match, including image URLs, legality, and purchase links.

## Environment Variables

- `ALLOWED_ORIGINS`: comma-separated list of CORS origins allowed by the backend
- `RATE_LIMIT_REQUESTS`: maximum requests per client IP per window (default: `30`)
- `RATE_LIMIT_WINDOW`: time window in seconds (default: `60`)

## Docker

Build the image:

```bash
docker build -t scryfall-search .
```

Run the container:

```bash
docker run --rm -p 8000:8000 scryfall-search
```

The container exposes the FastAPI app on port `8000`.

## Heroku / Procfile

The project includes a `Procfile` for platforms like Heroku:

```text
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

## Notes

- The backend normalizes embeddings before FAISS search.
- The frontend uses Scryfall's public API to enrich result cards.
- The search index is stored in `scryfall_index.faiss` and must be present before starting the app.
- `launch.py` can build missing assets with `python base.py` if the required files are absent.

## Project Structure

- `base.py` — index builder and asset preparation
- `main.py` — FastAPI backend and request handling
- `launch.py` — startup helper and browser launcher
- `index.html` — frontend UI and search experience
- `requirements.txt` — Python dependencies
- `Dockerfile` — container build instructions
- `Procfile` — deployment command for Heroku-style hosting
- `id_mapping.json` — saved card metadata mapping
- `scryfall_index.faiss` — vector index for image search
- `clip-model/` — local CLIP model files
- `images/` — downloaded card artwork cache

