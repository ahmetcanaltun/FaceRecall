"""
The dependency-free core: InsightFace model + persistent enrolled gallery + matching.

Rather than re-embedding every reference photo on each run, people are enrolled once into a
single human-readable JSON file (`data/gallery.json`), storing multiple embeddings per person
plus per-embedding metadata (source photo, detection score, face size, quality flags,
enrollment date), the raw material for threshold calibration and contamination checks
(see `experiments/threshold_calibration.py` and `experiments/holdout_evaluation.py`).

Storage format (user-chosen: single JSON file):

    {
      "meta":   {"backend": "insightface/buffalo_l", "dim": 512, "updated": "..."},
      "people": {
        "ana_maria_silva": {
          "embeddings": [
            {"src": "ana_maria_silva_0.jpg", "emb": [512 floats],
             "det_score": 0.82, "face_px": 240, "date": "2026-07-03", "flags": []}
          ]
        }, ...
      }
    }

Quality gates (from face-recognition-system-builder's enrollment decision tree) are applied
as *flags*, not hard rejects, data is scarce here (sometimes 1 photo/person), so we document
weak enrollments per person rather than silently dropping them. The only hard skip is a photo
where no face is detected at all.

`enroll()` is the single "add a person" path, the app's labeling / Wikimedia flows all call
it; there is not a second enrollment code path.

Library module: run everything through the app (`streamlit run src/app.py`). To rebuild the
gallery from data/reference_photos/ (rare maintenance), see the one-liner in the README.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
from pathlib import Path

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
BACKEND = "insightface/buffalo_l"
EMB_DIM = 512
DEFAULT_GALLERY = "data/gallery.json"
DEFAULT_PHOTOS = "data/reference_photos"

# Calibrated cosine-similarity cutoff for "same person" on buffalo_l embeddings (backend-
# specific, revisit if the backend ever changes). Photo pairs separate cleanly around it and
# on video it sits at the bottom edge of the genuine cluster; see
# experiments/threshold_calibration.py and results/threshold_calibration.json.
DEFAULT_THRESHOLD = 0.40

# Enrollment quality gates, recorded as flags, not hard rejects (scarce data).
MIN_CONFIDENCE = 0.7  # det_score below this -> "low_confidence"
MIN_FACE_PX = 80  # min(w,h) below this -> "small_face"
MAX_FACE_FRACTION = 0.5  # face wider/taller than this share of the image -> "large_face"

# A reference photo is an image someone handed us as "this contains the face",
# so unlike the video scan (det_thresh 0.5, tuned for noisy frames) we retry a failed
# detection once at this lower threshold before giving up. Rescued 39/47 previously
# un-enrollable video crops on the 2026-07-16 scan; the record gets a "weak_detection" flag
# (and, for video crops, review_unknowns.preflight's >=0.6 consistency check still guards
# against latching onto the wrong face).
RETRY_DET_THRESH = 0.2


# ---------------------------------------------------------------------------
# math helpers
# ---------------------------------------------------------------------------
def l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2(a), l2(b)))


# ---------------------------------------------------------------------------
# model (lazy, only import/spin up InsightFace when we actually need to embed,
# so gallery I/O, matching, and the tests stay usable without it installed)
# ---------------------------------------------------------------------------
_APP = None


def get_app():
    """Shared InsightFace detector/embedder. Prefers CoreML (Apple GPU/ANE, ~4.7x faster
    than CPU on this M1 Pro, with effectively identical embeddings: CPU-vs-CoreML parity
    ~0.9998 cosine, negligible vs the 0.42 separation margin) and falls back to CPU where
    CoreML isn't available (e.g. non-macOS). Same backend/model either way, this is an
    execution-provider choice, not a backend change."""
    global _APP
    if _APP is None:
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        if "CoreMLExecutionProvider" in ort.get_available_providers():
            providers, ctx = ["CoreMLExecutionProvider", "CPUExecutionProvider"], 0
        else:
            providers, ctx = ["CPUExecutionProvider"], -1
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=ctx, det_size=(640, 640))
        _APP = app
    return _APP


# ---------------------------------------------------------------------------
# enrollment
# ---------------------------------------------------------------------------
def _today() -> str:
    return _dt.date.today().isoformat()


def embed_photo(path: Path) -> dict | None:
    """Embed the most confident face in one photo, returning a gallery record (or None
    if no face is found / the image is unreadable). Records quality flags rather than
    rejecting weak faces."""
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        return None
    app = get_app()
    faces = app.get(img)
    weak_detection = False
    if not faces:  # retry once at the reference-photo threshold (see RETRY_DET_THRESH)
        old_thresh = app.det_model.det_thresh
        app.det_model.det_thresh = RETRY_DET_THRESH
        try:
            faces = app.get(img)
        finally:
            app.det_model.det_thresh = old_thresh
        weak_detection = True
    if not faces:
        return None

    face = max(faces, key=lambda d: d.det_score)  # reference photo = its most confident face
    x1, y1, x2, y2 = map(int, face.bbox)
    face_px = min(x2 - x1, y2 - y1)
    img_min = min(img.shape[0], img.shape[1])

    flags: list[str] = []
    if weak_detection:
        flags.append("weak_detection")  # only found by the RETRY_DET_THRESH second pass
    if face.det_score < MIN_CONFIDENCE:
        flags.append("low_confidence")
    if face_px < MIN_FACE_PX:
        flags.append("small_face")
    if img_min and max(x2 - x1, y2 - y1) > MAX_FACE_FRACTION * img_min:
        flags.append("large_face")
    if len(faces) > 1:
        flags.append("multiple_faces")  # ambiguous which subject, worth eyeballing

    return {
        "src": path.name,
        "emb": [round(float(v), 6) for v in face.normed_embedding],
        "det_score": round(float(face.det_score), 3),
        "face_px": int(face_px),
        "date": _today(),
        "flags": flags,
    }


def enroll(gallery: dict, name: str, image_paths: list[Path], *, verbose: bool = True) -> int:
    """Add embeddings for `name` from the given images into `gallery` (in place).
    Returns the number of embeddings added. This is the single enrollment entry point
    reused by the photo-folder build, the app's labeling flow, and the Wikimedia add-person
    flow."""
    person = gallery["people"].setdefault(name, {"embeddings": []})
    added = 0
    for p in sorted(image_paths):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        rec = embed_photo(p)
        if rec is None:
            if verbose:
                print(f"    skip (no face / unreadable): {p.name}")
            continue
        person["embeddings"].append(rec)
        added += 1
        if verbose:
            flagstr = f"  [{', '.join(rec['flags'])}]" if rec["flags"] else ""
            print(f"    + {p.name}  det={rec['det_score']:.2f}  face={rec['face_px']}px{flagstr}")
    return added


def build_from_photos(photos_root: str) -> dict:
    """Fresh gallery from data/reference_photos/<person>/*.<img>."""
    gallery = new_gallery()
    for pdir in sorted(p for p in Path(photos_root).iterdir() if p.is_dir()):
        print(f"  {pdir.name}:")
        enroll(gallery, pdir.name, list(pdir.iterdir()))
    touch(gallery)
    return gallery
