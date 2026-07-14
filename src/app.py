"""
The app: recognize the people in a video and teach it anyone it doesn't know.
A multi-page Streamlit app (`st.navigation`) over the library modules, the pass/clustering
(`collect_unknowns.collect`) and the labeling cards + safeguards
(`review_unknowns.render_manifest`) each live in exactly one place; the app only orchestrates.

Pages (grouped in the sidebar):
    Videos / Recognize    paste a YouTube link (or pick/upload) -> it downloads and scans in
                          one step; every face is matched, clustered into apparent people,
                          and decided recognized/unknown per cluster with the adopted 3-of-5
                          temporal rule at the calibrated 0.40 threshold. Name an unknown ->
                          it's enrolled through the single gallery.enroll() path (contamination
                          / mis-crop guarded); Ignore -> skipped.
    Videos / Live + learn one continuous flow: paste a link -> it streams live (no download) with
                          recognition boxes -> press Stop -> it names whatever it didn't recognize
                          (the same collect/finalize path + review cards as Recognize, driven live).
    People / Add person   type a name -> candidate faces from Wikimedia (wiki_faces) -> pick
                          1-2 -> enrolled (same guarded path). No manual photo hunting.
    People / Gallery      browse enrolled people (summary metrics + per-person cards) and
                          rename / delete / merge / drop a bad reference photo.

Naming a face updates data/gallery.json immediately, so the very next scan already knows them.

Run it (opens in your browser):
    .venv/bin/python -m streamlit run src/app.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

import collect_unknowns as C
import gallery as G
import live_recognition as L  # per-frame box/label drawing for the live view
import review_unknowns as R
import video_library as VL  # download + tidy naming + title index for the video picker
import wiki_faces as W  # Wikimedia face-candidate search for the "add person" page

VIDEOS_DIR = Path("data/videos")
UPLOAD_DIR = VIDEOS_DIR / "uploads"
GALLERY = "data/gallery.json"
PHOTOS_ROOT = "data/reference_photos"
OUT_ROOT = "results/unknowns"
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


@st.cache_resource
def get_app():
    return G.get_app()  # shared InsightFace detector/embedder, loaded once per session


def save_upload(uploaded) -> Path:
    """Persist an uploaded video under data/videos/uploads/ (git-ignored). Skips the rewrite
    if the same file is already there (this runs on every Streamlit rerun)."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / uploaded.name
    data = uploaded.getbuffer()
    if not dest.exists() or dest.stat().st_size != len(data):
        dest.write_bytes(data)
    return dest


def run_scan(video_path, gallery_path, threshold, cluster_sim, interval):
    """Scan a video, write its manifest, remember it in session_state. Returns (stats,
    recognized_people). Shared by 'Find faces' and the 'Re-scan to verify' button."""
    known = G.embeddings_from(gallery_path)
    if not known:
        st.error("The gallery is empty, build or enroll into it first.")
        st.stop()
    bar = st.progress(0.0, text="starting...")
    with st.spinner("Scanning the video..."):
        clusters, stats = C.collect(
            get_app(),
            known,
            video_path,
            threshold=threshold,
            cluster_sim=cluster_sim,
            interval=interval,
            progress=lambda frac, text: bar.progress(frac, text=text),
        )
        mpath = C.write_manifest(
            clusters, video_path, threshold=threshold, cluster_sim=cluster_sim, out_root=OUT_ROOT
        )
    bar.empty()
    recognized = sorted({c["candidate"] for c in clusters if c["recognized"]})
    st.session_state["manifest_path"] = str(mpath)
    st.session_state["manifest_video"] = str(video_path)
    st.session_state["scan_recognized"] = recognized
    st.session_state["scan_unknown"] = stats["unknown"]
    return stats, recognized


def _enroll_wiki_pick(
    cand, name, gallery_path, photos_root, threshold, override
) -> tuple[int, str | None]:
    """Enroll one Wikimedia crop as `name` through the single gallery.enroll() path (via
    review_unknowns.enroll_crop), guarding against mislabels:
    if the face is >= threshold-close to a different already-enrolled person it's refused
    until the reviewer overrides. Returns (embeddings_added, warning_or_None)."""
    crop = Path(cand["crop_path"])
    rec = G.embed_photo(crop)
    if rec is None:
        return 0, f"{cand['title']}: no face could be re-detected, skipped."
    if Path(gallery_path).exists() and not override:
        others = {k: v for k, v in G.load_embeddings(gallery_path).items() if k != name}
        if others:
            who, sim = G.match(np.asarray(rec["emb"], dtype=np.float32), others)
            if sim >= threshold:
                return 0, (
                    f"{cand['title']}: {sim:.2f}-close to **{R.pretty(who)}**; "
                    f"possible mislabel. Tick *Add anyway* if you're sure."
                )
    added = R.enroll_crop(crop, name, gallery_path, photos_root)
    if added == 0:
        return 0, f"{cand['title']}: no face re-detected on enroll, skipped."
    return added, None
