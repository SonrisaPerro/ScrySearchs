"""Smoke checks for a running ScrySearchs server.

    python check.py                                        # local server on port 8000
    python check.py https://scryfallsearch.up.railway.app  # the live site

Exits non-zero if anything fails. The weekly refresh runs this against the freshly
built index before publishing it, so a broken index never reaches the site.
"""
import io
import sys
import time

import requests
from PIL import Image

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
SCRYFALL = {"User-Agent": "ScrySearchs-check/1.0", "Accept": "*/*"}
failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}")
    if not ok:
        failures.append(name)


def scryfall_card(name):
    time.sleep(0.1)  # Scryfall asks for 50-100 ms between API calls
    r = requests.get("https://api.scryfall.com/cards/named", params={"exact": name}, headers=SCRYFALL, timeout=30)
    r.raise_for_status()
    return r.json()


def jpeg(url):
    img = Image.open(io.BytesIO(requests.get(url, headers=SCRYFALL, timeout=30).content)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def image_search(data, **filters):
    return requests.post(f"{BASE}/search", params=filters, files={"file": ("q.jpg", data, "image/jpeg")},
                         data={"leniency": "2"}, timeout=120)


def text_search(query, **filters):
    return requests.post(f"{BASE}/search/text", params=filters, data={"query": query}, timeout=120)


def lookup(matches):
    """Scryfall's own data for the matched cards, to check filters against the source of truth."""
    ids = [{"id": m["scryfall_id"]} for m in matches]
    cards = {}
    for start in range(0, len(ids), 75):
        time.sleep(0.1)
        r = requests.post("https://api.scryfall.com/cards/collection", json={"identifiers": ids[start:start + 75]},
                          headers=SCRYFALL, timeout=30)
        r.raise_for_status()
        cards.update({c["id"]: c for c in r.json()["data"]})
    return [cards[m["scryfall_id"]] for m in matches if m["scryfall_id"] in cards]


def type_line(card, match):
    face = next((f for f in card.get("card_faces", []) if f.get("illustration_id") == match["illustration_id"]), card)
    return (face.get("type_line") or card.get("type_line", "")).lower()


def main():
    r = requests.get(f"{BASE}/health", timeout=120)  # generous: a sleeping server takes a few seconds to wake
    check("server is up", r.ok and r.json().get("status") == "ok", f"HTTP {r.status_code}")

    # Image search: exact art at #1, labelled "Same art" by the page's rule (>= 0.86 and 0.025 ahead of #2).
    flare = scryfall_card("Final Flare")
    m = image_search(jpeg(flare["image_uris"]["art_crop"])).json()["matches"]
    same_art = m[0]["similarity_score"] >= 0.86 and m[0]["similarity_score"] - m[1]["similarity_score"] >= 0.025
    check("art upload finds the card at #1", m[0]["name"] == "Final Flare", m[0]["name"])
    check("...and would be labelled 'Same art'", same_art, f"{m[0]['similarity_score']} vs {m[1]['similarity_score']}")

    # A screenshot of the whole card should match through the art-box crop.
    elves = scryfall_card("Llanowar Elves")
    m = image_search(jpeg(elves["image_uris"]["normal"])).json()["matches"]
    rank = next((i + 1 for i, x in enumerate(m) if x["name"] == "Llanowar Elves"), None)
    check("full-card screenshot finds the card in the top 5", rank is not None and rank <= 5, f"rank {rank}")

    # Text search: loose on purpose, so ordinary index updates don't trip it.
    m = text_search("a goblin riding a bomb").json()["matches"]
    check("text search returns 30 results", len(m) == 30, len(m))
    check("text search scores look healthy", m[0]["similarity_score"] >= 0.27, m[0]["similarity_score"])
    check("goblin description finds a Goblin in the top 3", any("goblin" in x["name"].lower() for x in m[:3]),
          ", ".join(x["name"] for x in m[:3]))
    m = text_search("a quiet snowy forest at night").json()["matches"]
    check("snowy-forest description finds a Forest in the top 5", any("forest" in x["name"].lower() for x in m[:5]),
          ", ".join(x["name"] for x in m[:5]))

    # More like this.
    r = requests.get(f"{BASE}/similar", params={"illustration_id": flare["illustration_id"], "leniency": 2}, timeout=120).json()
    check("'more like this' returns 30 others",
          len(r["matches"]) == 30 and all(x["illustration_id"] != flare["illustration_id"] for x in r["matches"]))

    # Filters, verified against Scryfall rather than our own stored flags.
    combo = text_search("a dragon over a burning city", colors="UR", types="creature", format="modern").json()["matches"]
    colorless = text_search("a dragon over a burning city", colors="C").json()["matches"]
    cards = lookup(combo + colorless)
    combo_cards, colorless_cards = cards[:len(combo)], cards[len(combo):]
    check("filters still fill 30 results", len(combo) == 30 and len(colorless) == 30, f"{len(combo)}, {len(colorless)}")
    check("blue-red + creature + modern are all obeyed", all(
        set(c["color_identity"]) <= {"U", "R"} and c["legalities"]["modern"] == "legal" and "creature" in type_line(c, x)
        for c, x in zip(combo_cards, combo)))
    check("colourless-only is obeyed", all(not c["color_identity"] for c in colorless_cards))

    # Alchemy/Arena-only art is an oddity; art also printed on paper shows as the paper card.
    kemba = jpeg(scryfall_card("Kemba's Outfitter")["image_uris"]["art_crop"])
    shown = image_search(kemba).json()["matches"]
    hidden = image_search(kemba, hide_oddities="true").json()["matches"]
    check("Alchemy-only art is found when oddities are shown", shown[0]["name"] == "Kemba's Outfitter", shown[0]["name"])
    check("...and hidden with the oddities", all(x["name"] != "Kemba's Outfitter" for x in hidden))
    teferi = image_search(jpeg(scryfall_card("A-Teferi, Time Raveler")["image_uris"]["art_crop"]), hide_oddities="true").json()["matches"]
    check("rebalanced 'A-' art shows as the paper card", teferi[0]["name"] == "Teferi, Time Raveler", teferi[0]["name"])

    # Bad input is refused cleanly.
    check("unknown filter value -> 400", text_search("x", colors="Q").status_code == 400)
    check("empty description -> 400", text_search("").status_code == 400)
    check("unknown artwork -> 404",
          requests.get(f"{BASE}/similar", params={"illustration_id": "nope"}, timeout=60).status_code == 404)
    check("21 MB upload -> 413", image_search(b"\xff" * (21 * 1024 * 1024)).status_code == 413)
    bomb = io.BytesIO()
    Image.new("1", (12000, 11000)).save(bomb, "PNG")  # 15 KB file claiming 132 megapixels
    r = requests.post(f"{BASE}/search", files={"file": ("b.png", bomb.getvalue(), "image/png")}, timeout=120)
    check("decompression bomb -> 413", r.status_code == 413)

    print(f"\n{'All checks passed.' if not failures else f'{len(failures)} failed: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # a crash mid-run is a failure too, just a less tidy one
        print(f"FAIL  check run crashed: {exc!r}")
        sys.exit(1)
