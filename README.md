# ScrySearchs

Reverse image search for Magic card art. Give it a picture and it finds the cards that look like it.

**Try it:** [scryfallsearch.up.railway.app](https://scryfallsearch.up.railway.app) (also mirrored at [sonrisaperro.github.io/ScrySearchs](https://sonrisaperro.github.io/ScrySearchs/))

## What it does

- **Search with an image.** Artwork, a screenshot of a card, a photo, a sketch. Screenshots of whole cards get the art cropped out automatically, since that's the only part the index knows about.
- **Search by description.** Type something like "a goblin riding a bomb" and it'll find art that fits.
- **More like this.** Every result has a button to go find art that looks like *that* card.
- **Printings & prices.** See every paper printing that uses the same art, cheapest first, with a link to buy it.

Text and "more like this" searches get their own link, so you can send someone exactly what you found.

## How it works

Every unique piece of card art on Scryfall gets run through a CLIP model (`clip-ViT-B-32`), which turns each image into a list of numbers describing what it looks like. Those all live in a FAISS index, and a search is just "which of these ~55,000 are closest to yours?"

- `POST /search` embeds your upload and returns the 30 closest artworks. Card-shaped uploads also get searched as just their art box.
- `POST /search/text` does the same with your description, using CLIP's text side. Text and images land in the same space, which is why this works at all.
- `GET /similar` searches from a card's stored numbers, so there's nothing to upload or process.
- The page (`index.html`) looks each match up on Scryfall for the card image, legality, and prices.

## Files

- `main.py`: the backend. Serves the page and the three searches.
- `assets.py`: where the index lives and which model to use. Both scripts read it from here so they can't drift apart.
- `base.py`: builds the index, or updates it with only the new art.
- `launch.py`: starts everything locally and opens your browser.
- `index_version.txt`: which GitHub Release holds the current index.
- `.github/workflows/refresh-index.yml`: the weekly job that keeps the index current.

## Running it yourself

```bash
pip install -r requirements.txt
python launch.py
```

First launch downloads the index (about 110 MB) and the CLIP model, so give it a minute.

## Keeping the index up to date

A GitHub Action runs every Monday. It grabs the current index, embeds whatever art Scryfall added that week, and if anything's new, publishes a release and bumps `index_version.txt`. That commit redeploys the site on its own.

To run it early: **Actions → Refresh card index → Run workflow**. Or locally:

```bash
python base.py          # only embed new art (a few seconds most weeks)
python base.py --full   # redo everything (about 30 minutes on CPU)
```

A local run only updates your copy. To ship it, make a release with both files and update `index_version.txt`.

## API

`POST /search`: multipart form with `file` (an image, 20 MB max) and optional `leniency` (how far off a match can be, default `0.25`).

`POST /search/text`: form field `query`, 1 to 200 characters. Leniency doesn't apply here. Text scores are all bunched together (roughly 0.23 to 0.34 across the whole top 30), so the page just shows rank.

`GET /similar?illustration_id=...&leniency=0.25`: art similar to a card already in the index, not counting itself, plus a `source` object saying which card that was. Unknown ids get a 404.

All three give back:

```json
{
  "matches": [
    {"scryfall_id": "...", "name": "...", "illustration_id": "...", "similarity_score": 0.9542, "api_link": "https://api.scryfall.com/cards/..."}
  ]
}
```

`similarity_score` is cosine similarity. Heads up: card art that's merely similar still scores 0.84 to 0.93, so a high number alone doesn't mean you found the card. The page only says "Same art" when the top result scores at least 0.86 *and* beats second place by at least 0.025.

## Affiliate links

Buy links go through Scryfall's TCGplayer partner links by default. They land on the right product page and send the credit to Scryfall, which feels fair since this whole thing runs on their data. To use your own TCGplayer affiliate account instead, set `IMPACT_AFFILIATE_REDIRECT` at the top of the script in `index.html` to your deep-link prefix (it ends in `?u=`).

## Deployment

Railway builds the `Dockerfile`. The image comes with CPU-only PyTorch, the model, and the current index already inside, so it starts up without downloading anything. The health check (`/health`) and which files trigger a redeploy are set in Railway's service settings. GitHub Pages hosts a copy of the page that talks to the Railway backend.

Environment variables:

- `ALLOWED_ORIGINS`: comma-separated list of sites allowed to call the backend.
- `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW`: searches per visitor per window (defaults to 30 a minute). Visitors are told apart by Railway's `X-Real-IP` header.

## Thanks

Card data and images come from [Scryfall](https://scryfall.com), who are great. ScrySearchs isn't affiliated with or endorsed by Wizards of the Coast.

Keeping this awake costs money. If you like it, you can [help keep the server on](https://ko-fi.com/sonriso).
