"""
The video scan: make a pass over a video and group every face into per-person clusters,
deciding recognized/unknown per cluster, the unknowns are then labeled by a human in the
review UI (`review_unknowns.py`, driven by the app).

Every detected face is matched against the enrolled gallery at the calibrated 0.40 threshold.
The recognized/unknown decision is made per cluster with the **3-of-5 temporal rule** (>=3 of
some 5 consecutive observations >= threshold), not per frame, so a single hot/cold frame can't
flip it, the rule was picked from a 16-rule sweep, reported in the project report. A person
seen for fewer than 3 sampled observations (~1.2 s) therefore can never be auto-recognized and
surfaces as unknown.

One stranger spans many consecutive frames, so detections are **greedily clustered by
embedding similarity** (one representative crop per apparent person) and the reviewer sees one
card per stranger, not one per frame. The clustering cutoff (`cluster_sim`, default 0.5) is a
heuristic, not a calibrated value: it is higher than the 0.40 recognition threshold because
same-clip frames of one person share lighting/camera and sit well above cross-domain matches.

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

DEFAULT_OUT_ROOT = "results/unknowns"  # where write_manifest files each video's crops+manifest

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


def process_frame(
    app,
    known: dict,
    frame: np.ndarray,
    ts: float,
    clusters: list[dict],
    *,
    cluster_sim: float = 0.5,
) -> list[dict]:
    """Detect + match every face in one frame and greedily assign each into `clusters`
    (in place, the same machinery `collect()` uses per sampled frame). Returns
    `detect_matches`' face list, enough for a caller to draw recognition boxes (or feed a
    tracker) without a second detection pass. Pure w.r.t. display: the live view uses this to
    both accumulate clusters and draw, so a live pass and a batch scan build identical
    clusters."""
    faces = detect_matches(app, known, frame)
    for f in faces:
        x1, y1, x2, y2 = f["bbox"]
        obs = {
            "t": round(ts, 2),
            "bbox": [x1, y1, x2, y2],
            "sims": {p: round(s, 4) for p, s in f["sims"].items()},
        }
        q = _quality(f["det_score"], f["face_px"])
        _assign(
            clusters,
            f["emb"],
            f["sim"],
            f["person"],
            ts,
            q,
            _crop(frame, (x1, y1, x2, y2)),
            cluster_sim,
            obs=obs,
        )
    return faces


def decide_cluster(obs: list[dict], known: dict, threshold: float) -> dict:
    """The adopted per-cluster decision, in one place. `obs` is a cluster's observation list
    ([{t, bbox, sims}, ...]); the cluster's `candidate` is the gallery person it scores highest
    against at its best moment, and it is `recognized` iff that person's score series passes
    the 3-of-5 temporal rule at `threshold`. Returns the four fields to store on the cluster.

    Called by `finalize` (batch scan / Live+learn's Stop) and per tick by the live view, which
    needs the same verdict mid-pass to label its boxes, one rule, one implementation."""
    candidate = max(known, key=lambda p: max(o["sims"][p] for o in obs))
    series = [o["sims"][candidate] for o in obs]
    temporal_sim = temporal_score(series)
    return {
        "candidate": candidate,
        "peak_sim": max(series),
        "temporal_sim": temporal_sim,
        "recognized": temporal_sim is not None and temporal_sim >= threshold,
    }


def finalize(clusters: list[dict], known: dict, threshold: float) -> dict:
    """Apply the per-cluster decision to already-clustered detections, sort unknowns-first, and
    return {recognized, unknown}. Split out of `collect()` so a live pass (which builds
    `clusters` incrementally via `process_frame`) can reach the same decision on Stop."""
    for c in clusters:
        c.update(decide_cluster(c["obs"], known, threshold))
    n_reco = sum(c["recognized"] for c in clusters)
    clusters.sort(key=lambda c: (c["recognized"], c["first_t"]))  # unknowns first
    return {"recognized": n_reco, "unknown": len(clusters) - n_reco}


def collect(
    app,
    known: dict,
    video,
    *,
    threshold: float = G.DEFAULT_THRESHOLD,
    cluster_sim: float = 0.5,
    interval: float = 0.4,
    progress=None,
) -> tuple[list[dict], dict]:
    """Make one sampled pass over `video`, matching every face against the `known` gallery
    ({person: (N,512)}) and grouping all faces (recognized and not) into per-person clusters
    by embedding similarity, one entry per apparent person. Returns (clusters, stats):
      * clusters: each {rep_emb, rep_sim, rep_closest, rep_quality, crop(ndarray), n,
                   first_t, last_t, obs, candidate, peak_sim, temporal_sim, recognized}.
                   `obs` is the full per-observation score series
                   [{t, bbox, sims: {person: cosine}}, ...] in time order; `candidate` is the
                   gallery person the cluster is closest to at its best moment, `peak_sim`
                   that best single-observation cosine, `temporal_sim` the aggregated
                   TEMPORAL_K-of-TEMPORAL_N score (None if the cluster is too brief), and
                   `recognized` the adopted per-cluster decision:
                   temporal_sim >= threshold (the 3-of-5 rule, a single hot frame no
                   longer decides). rep_* fields describe the
                   highest-quality detection (whose crop is saved) and are kept for
                   display/enrollment. Sorted unknown-first, then by first appearance.
      * stats: {sampled, detections, recognized, unknown}.
    Pure (no disk writes) so both the CLI and the Streamlit app can call it. `progress(frac,
    text)` is called periodically if given, so a UI can show a bar.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps * interval)))

    clusters: list[dict] = []
    n_det = sampled = idx = 0
    while True:
        if not cap.grab():
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                sampled += 1
                faces = process_frame(
                    app, known, frame, idx / fps, clusters, cluster_sim=cluster_sim
                )
                n_det += len(faces)
                if progress and total:
                    progress(min(idx / total, 1.0), f"scanned {sampled} frames")
        idx += 1
    cap.release()
    if progress:
        progress(1.0, f"scanned {sampled} frames")
    stats = finalize(clusters, known, threshold)
    return clusters, {"sampled": sampled, "detections": n_det, **stats}


# The adopted decision rule: a cluster is recognized iff >=K of some N consecutive observations
# score >= threshold. At the ~0.4 s sampling interval that is a ~2 s window needing ~1.2 s of
# presence, so a single hot frame can't flip the decision either way.
TEMPORAL_K, TEMPORAL_N = 3, 5


def temporal_score(series: list[float], k: int = TEMPORAL_K, n: int = TEMPORAL_N) -> float | None:
    """Aggregated score under the k-of-n rule: the best value v such that some window of
    <=n consecutive observations has >=k observations >= v (= max over windows of the k-th
    largest value in the window). None if the series is shorter than k, too brief to ever
    pass. Compare the result against the recognition threshold."""
    m = len(series)
    if m < k:
        return None
    windows = [series] if m <= n else [series[i : i + n] for i in range(m - n + 1)]
    return max(sorted(w, reverse=True)[k - 1] for w in windows)


# ---------------------------------------------------------------------------
# ignore list, persists across re-scans (in results/unknowns/<video_id>/ignored.json),
# so a face the user ignored once isn't asked about again on the next scan.
# ---------------------------------------------------------------------------
def _ignored_path(out_dir) -> Path:
    return Path(out_dir) / "ignored.json"


def load_ignored(out_dir) -> list[np.ndarray]:
    p = _ignored_path(out_dir)
    if not p.exists():
        return []
    return [np.asarray(e, dtype=np.float32) for e in json.loads(p.read_text()).get("embs", [])]


def add_ignored(out_dir, emb) -> None:
    """Remember one ignored face's embedding (unit-norm) so future scans auto-skip it."""
    p = _ignored_path(out_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(p.read_text()) if p.exists() else {"embs": []}
    data["embs"].append([round(float(v), 6) for v in emb])
    p.write_text(json.dumps(data))


def clear_ignored(out_dir) -> int:
    """Forget all ignored faces for this video. Returns how many were forgotten."""
    p = _ignored_path(out_dir)
    n = len(load_ignored(out_dir))
    if p.exists():
        p.unlink()
    return n


def write_manifest(
    clusters: list[dict],
    video,
    *,
    threshold: float,
    cluster_sim: float,
    out_root: str = DEFAULT_OUT_ROOT,
) -> Path:
    """Persist `collect()`'s output as results/unknowns/<video_id>/{manifest.json, crop_*.jpg},
    the handoff the review UI reads. Returns the manifest path."""
    video_id = Path(video).stem
    out_dir = Path(out_root) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, c in enumerate(clusters):
        crop_name = f"crop_{i:03d}.jpg"
        cv2.imwrite(str(out_dir / crop_name), c["crop"])
        recognized = bool(c["recognized"])
        entries.append(
            {
                "id": i,
                "crop": crop_name,
                "n_frames": c["n"],
                "first_t": round(c["first_t"], 2),
                "last_t": round(c["last_t"], 2),
                "sim": round(c["peak_sim"], 3),  # best single-observation cosine to match
                "temporal_sim": (
                    None if c["temporal_sim"] is None else round(c["temporal_sim"], 3)
                ),  # k-of-n aggregated score
                "match": c["candidate"],  # closest gallery person
                "recognized": recognized,  # temporal_sim >= threshold (3-of-5)
                "emb": [round(float(v), 6) for v in c["rep_emb"]],  # for the contamination guard
                # recognized -> informational (correct if wrong); unknown -> needs a name
                "status": "recognized" if recognized else "pending",
            }
        )

    # Carry ignore decisions across re-scans: any unknown matching a previously-ignored face
    # (same video, cosine >= cluster_sim) is auto-marked skipped so it isn't asked about again.
    ignored = load_ignored(out_dir)
    if ignored:
        for e in entries:
            if e["recognized"]:
                continue
            q = np.asarray(e["emb"], dtype=np.float32)
            if any(float(np.dot(g, q)) >= cluster_sim for g in ignored):
                e["status"] = "skipped"

    manifest = {
        "video": str(video),
        "video_id": video_id,
        "threshold": threshold,
        "rule": f"{TEMPORAL_K}of{TEMPORAL_N}",
        "cluster_sim": cluster_sim,
        "created": _dt.date.today().isoformat(),
        "clusters": entries,
    }
    mpath = out_dir / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=1))
    return mpath


def _assign(clusters, emb, sim, person, ts, quality, crop, cluster_sim, obs) -> None:
    """Greedy online clustering of one face detection into `clusters` (in place)."""
    best_i, best_s = -1, cluster_sim
    for i, c in enumerate(clusters):
        s = float(np.dot(c["rep_emb"], emb))  # both unit-norm -> cosine
        if s >= best_s:
            best_i, best_s = i, s
    if best_i < 0:
        clusters.append(
            {
                "rep_emb": emb,
                "rep_sim": sim,
                "rep_closest": person,
                "rep_quality": quality,
                "crop": crop,
                "n": 1,
                "first_t": ts,
                "last_t": ts,
                "obs": [obs],
            }
        )
        return
    c = clusters[best_i]
    c["n"] += 1
    c["first_t"], c["last_t"] = min(c["first_t"], ts), max(c["last_t"], ts)
    c["obs"].append(obs)
    if quality > c["rep_quality"]:  # keep the highest-quality detection as the representative
        c.update(rep_emb=emb, rep_sim=sim, rep_closest=person, rep_quality=quality, crop=crop)
