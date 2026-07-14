"""
Face tracking between sparse detections, so a live overlay can draw at video rate.

The problem this solves: full detection + embedding is ~40-200 ms/frame, so running it on
every frame turns "live" playback into a slideshow. The fix (norfair, user decision
2026-07-14): run the expensive `collect_unknowns.detect_matches` pass only every ~0.4 s,
and in between let a Kalman filter carry each face's box forward, boxes follow faces at
full frame rate while recognition stays on the calibrated sampled cadence.

Identity per TRACK, not per frame: each track accumulates the per-person cosine map (`sims`)
of every detection assigned to it, and its label is decided with the same 3-of-5 temporal
rule the batch scan uses (`collect_unknowns.temporal_score`), a track flips from gray "?"
to a green name once >=3 of some 5 consecutive observations reach the threshold. This changes
nothing about the adopted decision mechanism (0.40 threshold, 3-of-5, greedy clustering for
the naming flow); the tracker is a display/labeling layer on top.

Consumers: `live_recognition.py` (the real-time cv2 window) and the app's Live + learn page.
"""

from __future__ import annotations

import numpy as np
from norfair import Detection, Tracker

import collect_unknowns as C
import gallery as G

# Ceiling on a track's time-to-live: how many consecutive update() calls it survives with no
# matching detection once fully warmed up. update() runs once per DISPLAYED frame while
# detections arrive only every ~0.4 s (~10-12 frames at 25-30 fps), so this must comfortably
# exceed one detection gap; ~45 frames ~ 1.5-2 s of absence. (A track's actual TTL is seeded
# from `period` on each matched detection and capped here, see FaceTracker.update.)
HIT_COUNTER_MAX = 45
# 1 - IoU distance cutoff for matching a detection to a track (norfair's "iou" distance):
# 0.7 accepts boxes overlapping by >= ~30%, forgiving enough for the box drift accumulated
# over a ~0.4 s prediction-only gap on news footage.
DISTANCE_THRESHOLD = 0.7


class FaceTracker:
    """Wraps a norfair Kalman tracker and decides each track's label with the 3-of-5 rule.

    Call `update(faces)` on a detection tick (faces = `detect_matches`/`process_frame`
    output) and `update(None)` on every in-between frame; both return the current tracks
    as drawable dicts: {id, bbox, person, sim, temporal_sim, recognized}.
    """

    def __init__(self, threshold: float = G.DEFAULT_THRESHOLD):
        self.threshold = threshold
        self.tracker = Tracker(
            distance_function="iou",
            distance_threshold=DISTANCE_THRESHOLD,
            hit_counter_max=HIT_COUNTER_MAX,
            initialization_delay=0,  # show a new face immediately (as "?" until the rule passes)
        )
        self._history: dict[int, list[dict[str, float]]] = {}  # track id -> [sims per obs]

    def update(self, faces: list[dict] | None = None, *, period: int = 1) -> list[dict]:
        """Advance one displayed frame. `faces` is this frame's detection output, or None for
        a frame with no detection pass (the Kalman filter predicts the boxes forward).
        On detection ticks pass `period` = the number of displayed frames per detection,
        norfair seeds/refreshes each track's time-to-live from it, so without this a track
        created from sparse detections dies before the next one arrives."""
        dets = []
        if faces:
            for f in faces:
                x1, y1, x2, y2 = f["bbox"]
                dets.append(Detection(points=np.array([[x1, y1], [x2, y2]], dtype=float), data=f))
        tracked = self.tracker.update(dets, period=period)

        fresh = {id(d.data) for d in dets}  # faces actually detected this frame
        out: list[dict] = []
        live_ids: set[int] = set()
        for t in tracked:
            live_ids.add(t.id)
            hist = self._history.setdefault(t.id, [])
            f = t.last_detection.data if t.last_detection is not None else None
            if f is not None and id(f) in fresh:
                hist.append(f["sims"])  # one observation per detection tick, like the scan
            person, temporal = self._decide(hist)
            (x1, y1), (x2, y2) = t.estimate
            out.append(
                {
                    "id": t.id,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "person": person,
                    "sim": hist[-1].get(person, 0.0) if hist else 0.0,
                    "temporal_sim": temporal,
                    "recognized": temporal is not None and temporal >= self.threshold,
                }
            )
        # Drop the score history of tracks norfair no longer reports (dead tracks).
        self._history = {k: v for k, v in self._history.items() if k in live_ids}
        return out

    def _decide(self, hist: list[dict[str, float]]) -> tuple[str, float | None]:
        """The track's best gallery candidate + its 3-of-5 temporal score (None while the
        track is too brief for the rule), the same decision the batch scan's finalize()
        makes per cluster, applied per track."""
        if not hist:
            return "", None
        people = {p for h in hist for p in h}
        if not people:
            return "", None
        candidate = max(people, key=lambda p: max(h.get(p, 0.0) for h in hist))
        series = [h.get(candidate, 0.0) for h in hist]
        return candidate, C.temporal_score(series)


def draw_tracks(frame, tracks: list[dict], *, show_unknown: bool = True) -> int:
    """Draw each track's box + label on `frame` in place (green once the 3-of-5 rule passes,
    gray "?" before that / for strangers). Returns the number of recognized tracks."""
    import cv2

    import live_recognition as L  # lazy: L imports tracking, so a top-level import would cycle

    hits = 0
    for t in tracks:
        if not t["recognized"] and not show_unknown:
            continue
        x1, y1, x2, y2 = t["bbox"]
        if t["recognized"]:
            color = (0, 220, 0)
            label = f"{L.short_name(t['person'])} {t['sim']:.2f}"
            hits += 1
        else:
            color = (150, 150, 150)
            label = f"? {t['sim']:.2f}" if t["sim"] else "?"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        L.draw_label(frame, x1, y1, label, color)
    return hits
