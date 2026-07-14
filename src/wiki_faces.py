"""
"Add a person" helper: given a name, fetch candidate face photos from Wikimedia, the
Wikipedia lead portrait + Commons file-search hits, so the reviewer can pick one or two
good faces and enroll them, instead of hunting the web and copying files by hand.

Why Wikimedia and not a generic web image search: this project avoids
watermarked stock (Getty/AFP) and prefers official / rights-clear sources. Wikimedia
Commons is free-licensed, has an official API (no scraping), and is an excellent match for this
project's domain, presidents, ministers and other public figures almost always have
Commons-licensed portraits. Each candidate carries its license string so the report can cite it.

The module only fetches and face-crops; enrollment still goes through the single
`gallery.enroll()` path (via review_unknowns.enroll_crop), so there is one enrollment code path.
Non-face files (signatures, logos, group shots with no usable face) are dropped by running the
same detector the gallery uses. Downloaded crops are cached under results/wiki_cache/<name>/
(git-ignored) and filed into data/reference_photos/<name>/ on enroll.

Library module: drives the app's "Add person" page. Network access required.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# A descriptive UA is required by the Wikimedia API etiquette.
USER_AGENT = "face-recognition-research/0.1 (research project; contact: local use)"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
CACHE_ROOT = Path("results/wiki_cache")
CROP_PAD = 0.35  # pad the detected box before cropping (room for clean re-alignment)
MIN_DET_SCORE = 0.55  # a candidate face must be at least this confident to be offered


def _api(base: str, params: dict) -> dict:
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{base}?{q}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def _lang_api(lang: str) -> str:
    return f"https://{lang}.wikipedia.org/w/api.php"


def search_candidates(name: str, *, limit: int = 12, langs=("tr", "en")) -> list[dict]:
    """Return up to `limit` candidate image records for `name`, most-canonical first:
    the Wikipedia lead portrait(s) then Commons file-search hits. Each record is
    {image_url, source_url, license, title}. Deduplicated by image_url. Network errors on
    any single source are swallowed so a partial result still comes back."""
    out: list[dict] = []
    seen: set[str] = set()

    def add(image_url, source_url, license_, title):
        if image_url and image_url not in seen:
            seen.add(image_url)
            out.append(
                {
                    "image_url": image_url,
                    "source_url": source_url,
                    "license": license_ or "?",
                    "title": title,
                }
            )

    # 1) Wikipedia lead image (the canonical portrait), try each language.
    for lang in langs:
        try:
            s = _api(
                _lang_api(lang),
                {
                    "action": "query",
                    "format": "json",
                    "redirects": 1,
                    "prop": "pageimages",
                    "piprop": "original",
                    "titles": name,
                },
            )
            for p in s.get("query", {}).get("pages", {}).values():
                src = p.get("original", {}).get("source")
                add(
                    src,
                    f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(name)}",
                    "Wikimedia Commons",
                    p.get("title", name),
                )
        except Exception:
            log.warning("%s.wikipedia lead-image lookup failed for %r", lang, name, exc_info=True)

    # 2) Commons file search, several more angles/photos.
    try:
        c = _api(
            COMMONS_API,
            {
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": name,
                "gsrnamespace": 6,
                "gsrlimit": limit,
                "prop": "imageinfo",
                "iiprop": "url|extmetadata",
                "iiurlwidth": 640,
            },
        )
        for f in c.get("query", {}).get("pages", {}).values():
            ii = (f.get("imageinfo") or [{}])[0]
            lic = ii.get("extmetadata", {}).get("LicenseShortName", {}).get("value")
            add(ii.get("url"), ii.get("descriptionurl"), lic, f.get("title", ""))
    except Exception:
        log.warning("Commons file search failed for %r", name, exc_info=True)

    return out[:limit]


def _download(url: str, dest: Path) -> Path | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as r:
            dest.write_bytes(r.read())
        return dest
    except Exception:
        log.warning("image download failed: %s", url, exc_info=True)
        return None


def _face_crop(app, img: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Detect the most confident face and return (padded_crop, det_score), or None if no
    usable face. Skips signatures/logos/landscapes (no face) and low-confidence detections."""
    faces = app.get(img)
    if not faces:
        return None
    face = max(faces, key=lambda d: d.det_score)
    if float(face.det_score) < MIN_DET_SCORE:
        return None
    x1, y1, x2, y2 = map(int, face.bbox)
    pw, ph = int((x2 - x1) * CROP_PAD), int((y2 - y1) * CROP_PAD)
    h, w = img.shape[:2]
    x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
    x2, y2 = min(w, x2 + pw), min(h, y2 + ph)
    return img[y1:y2, x1:x2].copy(), float(face.det_score)


def fetch_face_candidates(app, name: str, *, limit: int = 12) -> list[dict]:
    """Search -> download -> detect. Returns one record per image that has a usable face:
    {crop_path, det_score, license, source_url, title}. crop_path is a padded face crop
    cached under results/wiki_cache/<slug>/ (git-ignored), ready to enroll from. Images with
    no detectable face (signatures, logos) are dropped."""
    slug = urllib.parse.quote(name, safe="")
    cache = CACHE_ROOT / slug
    cache.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for i, cand in enumerate(search_candidates(name, limit=limit)):
        raw = _download(cand["image_url"], cache / f"src_{i:02d}")
        if raw is None:
            continue
        img = cv2.imread(str(raw))
        if img is None:
            continue
        got = _face_crop(app, img)
        if got is None:
            continue
        crop, det = got
        crop_path = cache / f"face_{i:02d}.jpg"
        cv2.imwrite(str(crop_path), crop)
        results.append(
            {
                "crop_path": str(crop_path),
                "det_score": round(det, 3),
                "license": cand["license"],
                "source_url": cand["source_url"],
                "title": cand["title"],
            }
        )
    return results
