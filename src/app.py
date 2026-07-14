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


def render_add_person(gallery_path, photos_root, threshold):
    """The 'Add person' page: type a name -> fetch candidate face photos from Wikimedia
    (rights-clear, official API, see wiki_faces) -> pick the ones that are really that person
    -> enroll them. No manual downloading/copying of reference photos."""
    st.subheader("Add a new person from Wikimedia")
    name_raw = st.text_input("Person's name", placeholder="e.g. İlkay Işık", key="wiki_name_in")
    if st.button("Search Wikimedia", disabled=not name_raw):
        with st.spinner(f"Searching Wikimedia for '{name_raw}'..."):
            st.session_state["wiki_cands"] = W.fetch_face_candidates(get_app(), name_raw)
        st.session_state["wiki_name"] = name_raw
        for k in list(st.session_state):  # clear stale pick checkboxes from a previous search
            if k.startswith("wpick_"):
                st.session_state.pop(k)

    cands = st.session_state.get("wiki_cands")
    name = st.session_state.get("wiki_name")
    if cands is None:
        return
    if not cands:
        st.warning(
            f"No usable face photos found on Wikimedia for '{name}'. Try the full "
            "name as it appears on Wikipedia."
        )
        return

    norm = R.norm_name(name)
    existing = norm in G.load(gallery_path)["people"] if Path(gallery_path).exists() else False
    if existing:
        st.caption(f"**{R.pretty(norm)}** is already enrolled, selections add to them.")
    cols = st.columns(4)
    for i, c in enumerate(cands):
        with cols[i % 4], st.container(border=True):
            st.image(c["crop_path"], width="stretch")
            st.checkbox("use this one", key=f"wpick_{i}")
            st.caption(f"{c['license']} · det {c['det_score']:.2f}")
            st.markdown(f"[source]({c['source_url']})")

    override = st.checkbox("Add anyway (override the contamination warnings)", key="wiki_override")
    if st.button(f"Add selected as {R.pretty(norm)}", type="primary", disabled=not norm):
        picked = [c for i, c in enumerate(cands) if st.session_state.get(f"wpick_{i}")]
        if not picked:
            st.warning("Tick at least one face first.")
            return
        added_total, warnings = 0, []
        for c in picked:
            added, warn = _enroll_wiki_pick(c, norm, gallery_path, photos_root, threshold, override)
            added_total += added
            if warn:
                warnings.append(warn)
        for w in warnings:
            st.error(w)
        if added_total:
            st.success(f"Enrolled **{added_total}** as **{R.pretty(norm)}**.")
            st.session_state.pop("wiki_cands", None)  # done; clear the candidate grid
            st.rerun()


def _render_person_card(name, embs, gallery, gallery_path, photos_root):
    """One bordered person card: name + count, a thumbnail strip, and (behind an expander to
    keep the page clean) the rename / delete / drop-photo controls."""
    with st.container(border=True):
        top, act = st.columns([4, 1])
        flags = sum(len(e.get("flags", [])) for e in embs)
        top.markdown(f"#### {R.pretty(name)}")
        top.caption(
            f"{len(embs)} reference photo(s)" + (f" · {flags} quality flag(s)" if flags else "")
        )
        manage = act.toggle(
            "Manage", key=f"manage_{name}", help="Rename, delete, or drop a reference photo"
        )

        thumbs = st.columns(min(len(embs), 8) or 1)
        for i, e in enumerate(embs):
            f = Path(photos_root) / name / e["src"]
            with thumbs[i % len(thumbs)]:
                if f.exists():
                    st.image(str(f), width="stretch")
                if e.get("flags"):
                    st.caption(", ".join(e["flags"]))
                if manage and st.button("remove", key=f"rm_{name}_{e['src']}"):
                    G.delete_embedding(gallery, name, e["src"], photos_root)
                    G.touch(gallery)
                    G.save(gallery, gallery_path)
                    st.rerun()

        if manage:
            ren, rbtn, dele = st.columns([3, 1, 1])
            new_raw = ren.text_input(
                "Rename to",
                key=f"rn_{name}",
                placeholder=R.pretty(name),
                label_visibility="collapsed",
            )
            if rbtn.button("Rename", key=f"rnbtn_{name}"):
                nn = R.norm_name(new_raw)
                if nn and nn != name:
                    try:
                        G.rename_person(gallery, name, nn, photos_root)
                        G.touch(gallery)
                        G.save(gallery, gallery_path)
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
            if dele.button("Delete", key=f"del_{name}"):
                st.session_state[f"confirm_del_{name}"] = True
            if st.session_state.get(f"confirm_del_{name}"):
                st.warning(f"Delete **{R.pretty(name)}** and all their reference photos?")
                yes, no = st.columns(2)
                if yes.button("Yes, delete", key=f"delyes_{name}"):
                    G.delete_person(gallery, name, photos_root)
                    G.touch(gallery)
                    G.save(gallery, gallery_path)
                    st.session_state.pop(f"confirm_del_{name}", None)
                    st.rerun()
                if no.button("Cancel", key=f"delno_{name}"):
                    st.session_state.pop(f"confirm_del_{name}", None)
                    st.rerun()


def render_gallery(gallery_path, photos_root):
    """The gallery tab: a clean roster of enrolled people as cards (name, thumbnails), each with
    per-card management (rename / delete / drop a bad reference photo) behind a Manage toggle,
    plus a merge control. A summary strip up top reads at a glance."""
    if not Path(gallery_path).exists():
        st.info("No one enrolled yet, use Add person.")
        return
    gallery = G.load(gallery_path)
    people = gallery["people"]
    total = sum(len(p["embeddings"]) for p in people.values())
    flagged = sum(1 for p in people.values() for e in p["embeddings"] if e.get("flags"))

    a, b, c = st.columns(3)
    a.metric("People", len(people))
    b.metric("Reference photos", total)
    c.metric("Quality-flagged", flagged)

    if len(people) >= 2:
        with st.expander("Merge two people into one"):
            names = sorted(people)
            c1, c2, c3 = st.columns([2, 2, 1])
            keep = c1.selectbox("Keep", names, key="merge_keep", format_func=R.pretty)
            drop = c2.selectbox(
                "Merge this one in",
                [n for n in names if n != keep],
                key="merge_drop",
                format_func=R.pretty,
            )
            c3.markdown("<div style='height:1.7em'></div>", unsafe_allow_html=True)
            if c3.button("Merge", key="merge_go"):
                moved = G.merge_people(gallery, drop, keep, photos_root)
                G.touch(gallery)
                G.save(gallery, gallery_path)
                st.success(f"Merged {R.pretty(drop)} -> {R.pretty(keep)} ({moved} photos).")
                st.rerun()

    for name in sorted(people):
        _render_person_card(name, people[name]["embeddings"], gallery, gallery_path, photos_root)
