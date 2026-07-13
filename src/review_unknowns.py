"""
The human-in-the-loop labeling UI: name the faces the scan didn't recognize.

Reads a manifest produced by `collect_unknowns.py` (one card per unrecognized person in a
clip) and lets a human name each one. Naming a face runs it back through the same enrollment
path the gallery already uses (`gallery.enroll`, the only enrollment
code path), so "ask the user who it is, learn the answer, add it to the database" is literally
the enroll() call. This closes the project's core loop.

Three safeguards, all mapped from known face-recognition failure modes:
  * Contamination guard: before enrolling, the new face is matched against the existing
    gallery; if it's >= threshold-close to a different already-enrolled person, we refuse
    until the reviewer explicitly overrides.
  * Provenance: the crop is copied into data/reference_photos/<name>/ so rebuilding the
    gallery from photos reproduces this enrollment; nothing is enrolled from a temp-only file.
  * Audit log: every decision (enroll/skip, who, when) is appended to decisions.jsonl next
    to the manifest, for the KVKK/biometric-data trail the project report covers.

Library module: `render_manifest` is the entry point, driven by the app
(`streamlit run src/app.py`).
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import shutil
import unicodedata
from pathlib import Path

import numpy as np
import streamlit as st

import collect_unknowns as C  # ignore-list store (persists skips across re-scans)
import gallery as G  # enrollment + match live here


def norm_name(raw: str) -> str:
    """'İlkay Işık' -> 'ilkay_isik', match the existing snake_case person keys.
    NFKD splits each accented letter into base + combining marks and the ASCII step drops
    the marks, including the U+0307 that lower() leaves on 'İ'. Dotless 'ı' is a standalone
    letter with no decomposition (ASCII would drop it whole), so it's mapped by hand."""
    s = raw.strip().lower().replace("ı", "i")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


@st.cache_resource
def _app():
    return G.get_app()  # load InsightFace once for the whole session


def load_manifest(path: str) -> dict:
    return json.loads(Path(path).read_text())


def save_manifest(manifest: dict, path: str) -> None:
    Path(path).write_text(json.dumps(manifest, indent=1))


def log_decision(mdir: Path, record: dict) -> None:
    record = {"ts": _dt.datetime.now().isoformat(timespec="seconds"), **record}
    with open(mdir / "decisions.jsonl", "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


CONSISTENCY_MIN = 0.6  # re-detected crop face must still match the clustered face this closely
