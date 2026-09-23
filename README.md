# ScrySearchs

Reverse image search for Magic: The Gathering card art, built with FastAPI, FAISS, and CLIP embeddings.

Upload artwork, a screenshot of a card, a photo, or a sketch, or just describe the art in words, and get the cards whose art looks closest. From any result you can browse art that looks like it, or see every printing of that art with prices.

## How it works

- Every unique piece of card art on Scryfall is embedded with the `clip-ViT-B-32` model and stored in a FAISS inner-product index (`scryfall_index.faiss`), with `id_mapping.json` mapping each vector back to its card.
- `POST /search` embeds the upload and returns the 30 nearest artworks. Card-shaped uploads are also searched as their art box, since the index only contains art crops.
- `POST /search/text` embeds a description with CLIP's text encoder and searches the same index.
- `GET /similar` searches from a card's stored vector, so "more like this" needs no upload.
- The frontend (`index.html`) looks up each match on Scryfall's API for images and legality, and lists every printing of a matched art (Scryfall's `illustrationid:` search) with prices and TCGplayer links. Text and similar searches get shareable links (`?q=...`, `?similar=...`).

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

`POST /search/text` takes a form field `query` (1 to 200 characters). Leniency doesn't apply: text-to-image scores are compressed (about 0.23 to 0.34 for a whole top 30), so the page shows rank instead.

`GET /similar?illustration_id=...&leniency=0.25` returns art similar to an indexed artwork (excluding itself), plus a `source` object naming it. Unknown ids return 404.

All three return:

```json
{
  "matches": [
    {"scryfall_id": "...", "name": "...", "illustration_id": "...", "similarity_score": 0.9542, "api_link": "https://api.scryfall.com/cards/..."}
  ]
}
```

`similarity_score` is cosine similarity. Similar-looking card art alone scores 0.84 to 0.93, so the page only labels a result "Same art" when the top hit scores at least 0.86 and leads the next by at least 0.025.

## Affiliate links

Buy links use Scryfall's TCGplayer partner links by default (they land on the product page and credit Scryfall). To use your own TCGplayer affiliate account, set `IMPACT_AFFILIATE_REDIRECT` at the top of the script in `index.html` to your deep-link prefix ending in `?u=`.

## Deployment

Railway builds the `Dockerfile`. Its health check (`/health`) and watch patterns (which files trigger a redeploy) are set in the Railway service settings. The image installs CPU-only PyTorch and bakes in the model and the pinned index, so containers start without downloading anything. The page is also published on GitHub Pages, where it calls the Railway backend.

Environment variables:

- `ALLOWED_ORIGINS`: comma-separated CORS origins.
- `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW`: requests per client IP per window (defaults 30 per 60 s). The client IP comes from Railway's `X-Real-IP` header.
