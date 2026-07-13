"""
The video scan: make a pass over a video and group every face into per-person clusters,
deciding recognized/unknown per cluster, the unknowns are then labeled by a human in the
review UI (`review_unknowns.py`, driven by the app).

Every detected face is matched against the enrolled gallery using the calibrated 0.40
threshold. The recognized/unknown decision is made per cluster with the adopted **3-of-5
temporal rule** (>=3 of some 5 consecutive observations >= threshold), not per frame, a
single hot/cold frame can no longer flip the decision (see
`experiments/temporal_aggregation.py` for the rule sweep that chose it). Clusters that fail
the rule are the "unknowns" this project's goal is about ("when it sees a face it doesn't
know, ask who it is"); note a person seen for fewer than 3 sampled observations (~1.2 s) can
never be auto-recognized and will surface as unknown.

The same stranger appears across many consecutive frames, so we don't want to ask "who is
this?" dozens of times for one person. Sub-threshold detections are therefore **greedily
clustered by embedding similarity** (one representative crop per apparent person), so the
reviewer sees one card per stranger, not one per frame. The clustering cutoff
(`cluster_sim`, default 0.5) is a heuristic for "same face across frames of one clip",
higher than the 0.40 recognition threshold because same-clip same-person frames share
lighting/camera and sit well above cross-domain gallery matches. Flagged for confirmation,
not a calibrated value.

Output (handoff to the review cards):
    results/unknowns/<video_id>/
        manifest.json      one entry per unknown cluster (crop, times, closest match, emb)
        crop_000.jpg ...   padded face crop for the best detection in each cluster

Library module: driven by the app (`streamlit run src/app.py`).
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import cv2
import numpy as np

import gallery as G  # shared CoreML detector/embedder + gallery match

MIN_DET_SCORE = 0.5  # same as the recognition demo: reject the noisiest detections
MIN_FACE_PX = 24  # skip only very tiny faces; keep this low so the scan surfaces (nearly)
# every face the live view shows, small ones get a `small_face` flag on
# enroll rather than being dropped silently
CROP_PAD = 0.25  # expand the bbox by this fraction each side before cropping: enough
# for the crop to re-detect cleanly through enroll(), but tight enough
# to avoid pulling a neighbour's face into a crowded crop


def _quality(det_score: float, face_px: int) -> float:
    """Pick the representative detection for a cluster: confident and large is best for a
    crop we may enroll from."""
    return det_score * face_px


def _crop(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    pw, ph = int((x2 - x1) * CROP_PAD), int((y2 - y1) * CROP_PAD)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
    x2, y2 = min(w, x2 + pw), min(h, y2 + ph)
    return frame[y1:y2, x1:x2].copy()


def detect_matches(app, known: dict, frame: np.ndarray) -> list[dict]:
    """Detect + match every face in one frame (det-score / min-size filtered) without touching
    any cluster state, the shared detection step for both the batch scan (`process_frame`)
    and the tracked live overlay (`tracking.FaceTracker`). Each face is
    {bbox, person, sim, sims, emb, det_score, face_px}: `person`/`sim` the best gallery match,
    `sims` the full per-person cosine map (the temporal rule needs the whole series)."""
    faces: list[dict] = []
    for d in app.get(frame):
        if d.det_score < MIN_DET_SCORE:
            continue
        x1, y1, x2, y2 = map(int, d.bbox)
        face_px = min(x2 - x1, y2 - y1)
        if face_px < MIN_FACE_PX:
            continue
        emb = np.asarray(d.normed_embedding, dtype=np.float32)
        sims = G.match_all(emb, known)
        person, sim = max(sims.items(), key=lambda kv: kv[1])
        faces.append(
            {
                "bbox": (x1, y1, x2, y2),
                "person": person,
                "sim": sim,
                "sims": sims,
                "emb": emb,
                "det_score": float(d.det_score),
                "face_px": face_px,
            }
        )
    return faces
