# ScrySearchs

Reverse image search for Magic: The Gathering card art, built with FastAPI, FAISS, and CLIP embeddings.

Upload artwork, a screenshot of a card, a photo, or a sketch, and get the cards whose art looks closest.

## How it works

- Every unique piece of card art on Scryfall is embedded with the `clip-ViT-B-32` model and stored in a FAISS inner-product index (`scryfall_index.faiss`), with `id_mapping.json` mapping each vector back to its card.
- `POST /search` embeds the upload and returns the 30 nearest artworks. Card-shaped uploads are also searched as their art box, since the index only contains art crops.
- The frontend (`index.html`) looks up each match on Scryfall's API for images, legality, and TCGplayer links.

## Files

- `main.py`: FastAPI backend (`/search`, `/health`, and serves `index.html`).
- `assets.py`: shared paths, the model name, and the download of the index from GitHub Releases.
- `base.py`: builds or refreshes the index from Scryfall's `unique_artwork` bulk data.
- `launch.py`: starts the backend and opens the browser.
- `index_version.txt`: the GitHub Release tag holding the index the app uses.
- `.github/workflows/refresh-index.yml`: weekly job that embeds new artwork and publishes a new release.

## Running locally

```bash
pip install -r requirements.txt
python launch.py
```

On first start the backend downloads the index (about 110 MB) from the release named in `index_version.txt`, plus the CLIP model from Hugging Face.

## Updating the index

The weekly GitHub Action does this automatically: it downloads the current index, embeds only artwork that is new on Scryfall, and if anything changed, publishes a new release and commits the new tag to `index_version.txt`. That commit triggers a redeploy.

To run it by hand, use **Actions → Refresh card index → Run workflow**, or locally:

```bash
python base.py          # incremental: embed only new artwork
python base.py --full   # re-embed everything (about 30 minutes on CPU)
```

A local run only changes your local copy. To ship it, create a release with both files and update `index_version.txt`.

## API

`POST /search` takes a multipart form with `file` (an image, max 20 MB) and optional `leniency` (maximum cosine distance, default `0.25`).

```json
{
  "matches": [
    {"scryfall_id": "...", "name": "...", "similarity_score": 0.9542, "api_link": "https://api.scryfall.com/cards/..."}
  ]
}
```

`similarity_score` is cosine similarity: an exact art match scores above about 0.9, and photos and sketches usually land around 0.75 to 0.85.

## Deployment

Railway builds the `Dockerfile` (selected in `railway.json`). The image installs CPU-only PyTorch and bakes in the model and the pinned index, so containers start without downloading anything. The page is also published on GitHub Pages, where it calls the Railway backend.

Environment variables:

- `ALLOWED_ORIGINS`: comma-separated CORS origins.
- `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW`: requests per client IP per window (defaults 30 per 60 s). The client IP comes from Railway's `X-Real-IP` header.
