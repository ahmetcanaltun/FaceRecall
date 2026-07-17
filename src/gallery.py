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
        # Only the two models we actually use. buffalo_l ships 5; without this, app.get()
        # also runs 106-point landmarks, 3D landmarks and age/gender on every face, ~45 ms
        # of pure waste per frame on crowded footage (measured 2026-07-17: 106 -> 61 ms/frame
        # on the live window's per-frame path, identical detections and embeddings).
        app = FaceAnalysis(
            name="buffalo_l", providers=providers, allowed_modules=["detection", "recognition"]
        )
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


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
def new_gallery() -> dict:
    return {"meta": {"backend": BACKEND, "dim": EMB_DIM, "updated": _today()}, "people": {}}


def touch(gallery: dict) -> None:
    gallery["meta"]["updated"] = _today()


def save(gallery: dict, path: str) -> None:
    """Write the gallery as indented (human-readable) JSON, but keep each 512-float `emb`
    array on one line. Plain `indent=1` explodes every embedding to 512 lines, a 4-person
    gallery becomes ~16.7k lines, which makes the file unscannable and git diffs enormous.
    We pretty-print the structure with each emb swapped for a unique placeholder string, then
    splice the compact one-line arrays back in. Values are byte-for-byte the same embeddings."""
    import copy

    g = copy.deepcopy(gallery)
    compact: list[str] = []
    for person in g["people"].values():
        for rec in person["embeddings"]:
            compact.append(json.dumps(rec["emb"]))  # one-line array, exact values
            rec["emb"] = f"__EMB_{len(compact) - 1}__"  # unique placeholder json will quote
    text = json.dumps(g, indent=1)
    for i, arr in enumerate(compact):
        text = text.replace(f'"__EMB_{i}__"', arr)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: fill a temp file in the same dir, then os.replace() onto the target. A plain
    # write_text() truncates the file to 0 bytes before rewriting, so a reader landing in that
    # window (another Streamlit session/tab, a CLI run, or a save interrupted mid-write) sees an
    # empty file and json.load() dies with "Expecting value: line 1 column 1 (char 0)". replace()
    # is atomic on POSIX, so readers always see the whole old or whole new file, never empty.
    tmp = p.with_name(f"{p.name}.tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, p)


def load(path: str) -> dict:
    text = Path(path).read_text()
    if not text.strip():  # empty file, e.g. a save was interrupted (see save() note)
        raise ValueError(
            f"Gallery {path!r} is empty, a save was likely interrupted. Restore it from a "
            f"backup, or rebuild it from reference photos (see the README)."
        )
    g = json.loads(text)
    if g.get("meta", {}).get("backend") != BACKEND:
        print(
            f"  WARNING: gallery backend {g.get('meta', {}).get('backend')!r} != {BACKEND!r}; "
            f"embeddings are backend-specific and won't match."
        )
    return g


def load_embeddings(path: str) -> dict[str, np.ndarray]:
    """Load the gallery as {person: (N,512) float32 matrix} for fast matching, the form
    the video/annotate scripts want. Embeddings are stored already L2-normalized."""
    g = load(path)
    out: dict[str, np.ndarray] = {}
    for person, data in g["people"].items():
        embs = [r["emb"] for r in data["embeddings"]]
        if embs:
            out[person] = np.asarray(embs, dtype=np.float32)
    return out


def embeddings_from(path: str) -> dict[str, np.ndarray]:
    """Convenience loader for the video/annotate scripts: if `path` is a saved gallery
    .json, load it (fast, no re-embedding); if it's a directory of reference photos,
    build the embeddings in memory on the fly (legacy path). Missing file -> clear error."""
    p = Path(path)
    if p.is_dir():
        g = build_from_photos(str(p))
        return {
            person: np.asarray([r["emb"] for r in d["embeddings"]], dtype=np.float32)
            for person, d in g["people"].items()
            if d["embeddings"]
        }
    if not p.exists():
        raise SystemExit(
            f"Gallery {path!r} not found, enroll someone in the app "
            f"(People ▸ Add person), or rebuild it from reference photos "
            f"(see the README's gallery-rebuild one-liner)."
        )
    return load_embeddings(path)


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------
def match(emb: np.ndarray, embeddings: dict[str, np.ndarray]) -> tuple[str, float]:
    """Return (closest_person, max_cosine). `emb` is a raw embedding; stored embeddings
    are already unit-norm, so cosine = emb_hat · stored."""
    q = l2(np.asarray(emb, dtype=np.float32))
    best_person, best_sim = "", -1.0
    for person, mat in embeddings.items():
        s = float(np.max(mat @ q))  # max over that person's reference embeddings
        if s > best_sim:
            best_person, best_sim = person, s
    return best_person, best_sim


def match_all(emb: np.ndarray, embeddings: dict[str, np.ndarray]) -> dict[str, float]:
    """Cosine of `emb` against every gallery person (max over that person's references),
    not just the argmax, the temporal decision rule needs the full
    per-person score series, which match() throws away."""
    q = l2(np.asarray(emb, dtype=np.float32))
    return {person: float(np.max(mat @ q)) for person, mat in embeddings.items()}


# ---------------------------------------------------------------------------
# management (used by the Streamlit gallery tab), operate on the gallery dict
# in place; the caller is responsible for touch() + save(). `photos_root` (when
# given) keeps the on-disk reference_photos/<name>/ folders in sync.
# ---------------------------------------------------------------------------
def _person_dir(photos_root: str, name: str) -> Path:
    return Path(photos_root) / name


def delete_person(gallery: dict, name: str, photos_root: str | None = None) -> None:
    """Remove a person from the gallery (and their reference-photo folder, if given)."""
    gallery["people"].pop(name, None)
    if photos_root:
        d = _person_dir(photos_root, name)
        if d.exists():
            shutil.rmtree(d)


def delete_embedding(gallery: dict, name: str, src: str, photos_root: str | None = None) -> None:
    """Drop one reference embedding (by its source filename) from a person; delete the file
    too if `photos_root` is given. If it was the person's last embedding, remove the person."""
    person = gallery["people"].get(name)
    if not person:
        return
    person["embeddings"] = [e for e in person["embeddings"] if e["src"] != src]
    if photos_root:
        f = _person_dir(photos_root, name) / src
        if f.exists():
            f.unlink()
    if not person["embeddings"]:
        delete_person(gallery, name, photos_root)


def rename_person(gallery: dict, old: str, new: str, photos_root: str | None = None) -> None:
    """Rename a person. Raises if `new` already exists (use merge_people instead)."""
    if old not in gallery["people"]:
        raise ValueError(f"no such person: {old!r}")
    if new in gallery["people"]:
        raise ValueError(f"{new!r} already exists, merge instead of rename")
    gallery["people"][new] = gallery["people"].pop(old)
    if photos_root:
        src_dir, dst_dir = _person_dir(photos_root, old), _person_dir(photos_root, new)
        if src_dir.exists() and not dst_dir.exists():
            src_dir.rename(dst_dir)


def merge_people(gallery: dict, src: str, dst: str, photos_root: str | None = None) -> int:
    """Fold `src`'s embeddings into `dst`, then delete `src`. Reference-photo files are moved
    into `dst`'s folder under fresh, non-colliding names (each embedding's `src` field updated
    to match). Returns the number of embeddings moved."""
    if src == dst:
        return 0
    people = gallery["people"]
    if src not in people or dst not in people:
        raise ValueError("both people must exist to merge")
    dst_embs = people[dst]["embeddings"]
    n0 = len(dst_embs)
    src_dir = _person_dir(photos_root, src) if photos_root else None
    dst_dir = _person_dir(photos_root, dst) if photos_root else None
    if dst_dir:
        dst_dir.mkdir(parents=True, exist_ok=True)
    for i, e in enumerate(people[src]["embeddings"]):
        if src_dir and dst_dir:
            old_file = src_dir / e["src"]
            new_name = f"{dst}_{n0 + i}{Path(e['src']).suffix.lower()}"
            if old_file.exists():
                shutil.move(str(old_file), str(dst_dir / new_name))
            e["src"] = new_name
        dst_embs.append(e)
    moved = len(people[src]["embeddings"])
    delete_person(gallery, src, photos_root)  # removes src + its (now-empty) folder
    return moved
