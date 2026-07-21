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
    People / Add person   type a name -> candidate faces from Wikimedia (wiki_faces) OR from
                          photos you upload -> pick 1-2 -> enrolled (same guarded path).
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
    """Enroll one candidate crop (Wikimedia or upload) as `name` through the single
    gallery.enroll() path (via
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


def _set_candidates(cands, name_raw):
    """Install a fresh candidate grid (from either source) and drop the previous grid's
    tick state, whose keys would otherwise select the wrong cards."""
    st.session_state["wiki_cands"] = cands
    st.session_state["wiki_name"] = name_raw
    for k in list(st.session_state):
        if k.startswith("wpick_"):
            st.session_state.pop(k)


def render_add_person(gallery_path, photos_root, threshold):
    """The 'Add person' page: type a name -> get candidate face photos, either from Wikimedia
    (rights-clear, official API, see wiki_faces) or from files the user uploads -> pick the
    ones that are really that person -> enroll them. Both sources produce the same candidate
    records and enroll through the same guarded path, so only the sourcing differs."""
    st.subheader("Add a new person")
    name_raw = st.text_input("Person's name", placeholder="e.g. İlkay Işık", key="wiki_name_in")

    wiki_col, up_col = st.columns(2)
    with wiki_col:
        if st.button("Search Wikimedia", disabled=not name_raw):
            with st.spinner(f"Searching Wikimedia for '{name_raw}'..."):
                _set_candidates(W.fetch_face_candidates(get_app(), name_raw), name_raw)
    with up_col:
        files = st.file_uploader(
            "...or upload photos",
            type=[e[1:] for e in G.IMAGE_EXTS],
            accept_multiple_files=True,
            key="up_photos",
        )
        if st.button("Use uploaded photos", disabled=not (name_raw and files)):
            with st.spinner(f"Detecting faces in {len(files)} photo(s)..."):
                _set_candidates(
                    W.face_candidates_from_uploads(
                        get_app(), name_raw, [(f.name, f.getvalue()) for f in files]
                    ),
                    name_raw,
                )

    cands = st.session_state.get("wiki_cands")
    name = st.session_state.get("wiki_name")
    if cands is None:
        return
    if not cands:
        st.warning(
            f"No usable face found for '{name}'. From Wikimedia: try the full name as it "
            "appears on Wikipedia. From an upload: the face may be too small or too "
            "low-confidence to enroll from."
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
            if c["source_url"]:  # uploads have no source page
                st.markdown(f"[source]({c['source_url']})")
            else:
                st.caption(c["title"])

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


def _download_box():
    """The 'paste a YouTube link' control. Downloads via video_library (best quality up to
    1080p when ffmpeg is present) and auto-selects it. Optional start/end minutes download
    just that section, the usual case, since a person is on screen only briefly in a long clip."""
    url = st.text_input(
        "Paste a YouTube link", key="yt_url", placeholder="https://youtube.com/watch?v=..."
    )
    ffmpeg = VL.has_ffmpeg()
    c_s, c_e = st.columns(2)
    start_min = c_s.number_input(
        "From (min)",
        min_value=0.0,
        value=0.0,
        step=0.5,
        disabled=not ffmpeg,
        help=None if ffmpeg else "Install ffmpeg to download a section.",
    )
    end_min = c_e.number_input(
        "To (min, 0 = end)", min_value=0.0, value=0.0, step=0.5, disabled=not ffmpeg
    )
    if st.button("Download & recognize", type="primary", disabled=not url):
        bar = st.progress(0.0, text="starting...")

        def hook(d):
            if d.get("status") == "downloading" and d.get("total_bytes"):
                bar.progress(
                    min(d["downloaded_bytes"] / d["total_bytes"], 1.0), text="downloading..."
                )

        start = start_min * 60 if (ffmpeg and (start_min or end_min)) else None
        end = end_min * 60 if (ffmpeg and end_min) else None
        path = None
        try:
            with st.spinner("Downloading..."):
                path = VL.download(url, progress_hook=hook, start=start, end=end)
        except Exception as e:
            st.error(f"Download failed: {e}")
        bar.empty()
        if path is not None:
            # non-widget key, consumed at the top of main() before the picker is built, so it
            # can select the new video without the "modified after instantiated" error. rerun is
            # kept outside the try so its control-flow exception isn't swallowed as a failure.
            st.session_state["_pending_video"] = str(path)
            st.session_state["_auto_scan"] = True  # URL flow = one hamle: scan right after download
            st.rerun()


DEF_INTERVAL, DEF_THRESHOLD, DEF_CLUSTER = 0.5, G.DEFAULT_THRESHOLD, 0.50


def _settings() -> tuple[str, float, float, float]:
    """Sidebar 'Advanced' settings, shared by every page. Stable widget keys mean the values
    persist as you move between pages. Calibrated defaults are tucked away so the common flow
    stays 'paste a link -> recognize'."""
    with st.sidebar, st.expander("Advanced settings", expanded=False):
        gallery_path = st.text_input("Gallery file", GALLERY, key="cfg_gallery")
        interval = st.slider(
            "Sample every ... seconds", 0.2, 2.0, DEF_INTERVAL, 0.1, key="cfg_interval"
        )
        threshold = st.slider(
            "Recognition threshold",
            0.20,
            0.60,
            DEF_THRESHOLD,
            0.01,
            key="cfg_threshold",
            help="Calibrated value is 0.40. A person is recognized iff "
            ">=3 of some 5 consecutive samples reach this (the 3-of-5 "
            "rule), so someone seen <3 samples stays unknown.",
        )
        cluster_sim = st.slider(
            "Same-person grouping cutoff",
            0.30,
            0.80,
            DEF_CLUSTER,
            0.05,
            key="cfg_cluster",
            help="Heuristic for merging one stranger's frames into one card.",
        )
    return gallery_path, interval, threshold, cluster_sim


def _video_source() -> Path | None:
    """Render the video picker (paste-link / library / upload) plus a modest-size preview, and
    return the chosen video path (or None). Stable keys keep the selection across reruns."""
    vids = VL.library()
    options = ["- none -"] + [str(p) for p in vids]
    pending = st.session_state.pop("_pending_video", None)  # set by download / rename-to-people
    if pending and pending in options:
        st.session_state["video_pick"] = pending
    elif st.session_state.get("video_pick") not in options:
        st.session_state["video_pick"] = options[1] if vids else "- none -"

    _download_box()  # paste link -> download + auto-recognize
    lib_col, up_col = st.columns(2)
    with lib_col:
        picked = st.selectbox(
            "Choose from the library",
            options,
            key="video_pick",
            format_func=lambda s: s if s == "- none -" else VL.title_for(s),
        )
    with up_col:
        uploaded = st.file_uploader("...or upload a file", type=[e[1:] for e in VIDEO_EXTS])

    video_path: Path | None = None
    if uploaded is not None:
        video_path = save_upload(uploaded)
    elif picked != "- none -":
        video_path = Path(picked)
    if video_path is not None:
        prev, _spacer = st.columns([2, 3])  # keep the preview modest, not full-screen-wide
        prev.video(str(video_path))
    return video_path


def page_recognize() -> None:
    """Main page: find who's in a video and name anyone it doesn't recognize."""
    gallery_path, interval, threshold, cluster_sim = _settings()
    st.header("Recognize")
    video_path = _video_source()
    if video_path is None:
        return

    thorough = st.checkbox(
        "Thorough, every frame (slower, catches brief faces)",
        value=False,
    )
    scan_interval = 0.0 if thorough else interval  # 0 -> step 1 -> every frame
    have_scan = (
        st.session_state.get("manifest_path")
        and Path(st.session_state.get("manifest_path", "")).exists()
        and st.session_state.get("manifest_video") == str(video_path)
    )
    # URL flow: a fresh download set _auto_scan, so recognition runs immediately with no extra
    # click. Pick/upload sources still use the explicit button below.
    if st.session_state.pop("_auto_scan", False):
        stats, _ = run_scan(video_path, gallery_path, threshold, cluster_sim, scan_interval)
        have_scan = True
        st.success(
            f"Scanned {stats['sampled']} frames · {stats['detections']} faces seen · "
            f"**{stats['recognized']} recognized**, **{stats['unknown']} unknown**."
        )
    c_find, c_verify = st.columns(2)
    if c_find.button("Recognize faces", type="primary"):
        stats, _ = run_scan(video_path, gallery_path, threshold, cluster_sim, scan_interval)
        have_scan = True
        st.success(
            f"Scanned {stats['sampled']} frames · {stats['detections']} faces seen · "
            f"**{stats['recognized']} recognized**, **{stats['unknown']} unknown**."
        )
    if have_scan and c_verify.button("Re-scan to verify"):
        prev = set(st.session_state.get("scan_recognized", []))
        prev_unknown = st.session_state.get("scan_unknown")
        stats, recognized = run_scan(
            video_path, gallery_path, threshold, cluster_sim, scan_interval
        )
        newly = sorted(set(recognized) - prev)
        if newly:
            st.success(
                "Now recognized: "
                + ", ".join(R.pretty(n) for n in newly)
                + f"  ·  unknown {prev_unknown} -> {stats['unknown']}."
            )
        else:
            st.info(f"No change, still {stats['unknown']} unknown.")

    if have_scan:
        mpath = st.session_state["manifest_path"]
        manifest = R.load_manifest(mpath)
        ppl = sorted({c["match"] for c in manifest["clusters"] if c.get("recognized")})
        if ppl:
            label = ", ".join(R.pretty(p) for p in ppl)
            if st.button(f"Rename video to the people in it, {label}"):
                newp = VL.rename_to_people(video_path, ppl)
                st.session_state["_pending_video"] = str(newp)
                st.session_state["manifest_video"] = str(newp)
                st.rerun()
        R.render_manifest(mpath, gallery_path, PHOTOS_ROOT)


# ---------------------------------------------------------------------------
# Live + learn , a continuous flow: paste a link -> it streams and plays with
# recognition boxes (nothing is downloaded) -> press Stop -> it asks who every
# unrecognized face is. (Prototype.)
#
# No download: video_library.stream_url resolves the link to a direct media URL via yt-dlp
# (skip_download) and OpenCV reads frames straight off it. Streamlit can't interrupt a
# blocking while-loop with a button, so playback is a self-rerunning st.fragment: each tick
# runs the shared per-frame path (collect_unknowns.process_frame, same detect/match/cluster
# the batch scan uses) on one sampled frame, draws boxes, and accumulates clusters in
# session_state. Stop / end-of-stream runs collect_unknowns.finalize + write_manifest on what
# was seen, then the existing review cards (review_unknowns.render_manifest) name the unknowns.
# `ll_src` = what OpenCV opens (stream URL or local path); `ll_video` = a short id for the
# manifest folder. All state under "ll_*" session keys; the cap stays open across ticks/pause
# (streaming never needs to seek), released only on Stop / Restart / New source.
# ---------------------------------------------------------------------------
def _reset_live(src, vid_id, gallery_path) -> None:
    """(Re)initialise the live session for a source: reload the gallery, drop the frame position
    / accumulated clusters / tracker / any capture, and leave it paused at the start of `src`."""
    cap = st.session_state.pop("ll_cap", None)
    if cap is not None:
        cap.release()
    st.session_state.update(
        ll_src=str(src),
        ll_video=str(vid_id),
        ll_known=G.embeddings_from(gallery_path),
        ll_clusters=[],
        ll_pos=0,
        ll_playing=False,
        ll_review=None,
        ll_done=False,
        ll_last_frame=None,
    )


def _clear_live() -> None:
    """Tear the live session down entirely (back to the source picker)."""
    cap = st.session_state.pop("ll_cap", None)
    if cap is not None:
        cap.release()
    for k in (
        "ll_src",
        "ll_video",
        "ll_known",
        "ll_clusters",
        "ll_pos",
        "ll_playing",
        "ll_review",
        "ll_done",
        "ll_last_frame",
        "ll_title",
        "ll_fps",
    ):
        st.session_state.pop(k, None)


def _live_cap(src):
    """The persistent cv2.VideoCapture for the live session (kept across fragment reruns in
    session_state). For a stream URL this opens an HTTP read; it's reopened + re-seeked only if
    it was released/closed (normal play/pause keeps it open, so streaming never seeks)."""
    cap = st.session_state.get("ll_cap")
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(str(src))
        st.session_state["ll_cap"] = cap
        st.session_state["ll_fps"] = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if st.session_state.get("ll_pos"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state["ll_pos"])
    return cap


def _finish_live(vid_id, threshold, cluster_sim, *, done: bool) -> None:
    """Turn what the live pass has seen so far into a review manifest: finalize the clusters
    with the 3-of-5 rule, write crops + manifest.json under results/unknowns/<vid_id>/, then
    flip into review mode. `done` distinguishes reaching the stream's end from a manual Stop."""
    clusters = st.session_state.get("ll_clusters", [])
    known = st.session_state.get("ll_known") or {}
    C.finalize(clusters, known, threshold)
    mpath = C.write_manifest(
        clusters, vid_id, threshold=threshold, cluster_sim=cluster_sim, out_root=OUT_ROOT
    )
    cap = st.session_state.pop("ll_cap", None)
    if cap is not None:
        cap.release()
    st.session_state.update(ll_playing=False, ll_review=str(mpath), ll_done=done)


@st.fragment(run_every="0.15s")
def _live_fragment(threshold, cluster_sim, interval, show_unknown, width):
    """One playback tick, auto-rerun on a timer (so a Stop/Pause button, rendered outside the
    fragment, can still be clicked between ticks). Reads the next sampled frame off the open
    stream, accumulates clusters via the shared path, draws boxes, and advances. On end-of-
    stream it finalises and reruns the whole app into review mode. `width` caps the on-screen
    frame size (px) so the player isn't full-page-wide.

    one frame per tick: an in-tick playback loop (tried with the Kalman
    tracker, 2026-07-14) fights the fragment timer, reruns overlap the loop and playback
    stutters/jumps, so it was reverted. The tracked overlay lives in live_recognition.py."""
    if not st.session_state.get("ll_playing"):
        last = st.session_state.get("ll_last_frame")
        if last is not None:  # keep the paused frame on screen instead of a blank gap
            st.image(last, channels="BGR", width=width)
            st.caption("⏸ paused")
        return
    known = st.session_state.get("ll_known") or {}
    clusters = st.session_state.setdefault("ll_clusters", [])
    cap = _live_cap(st.session_state["ll_src"])
    fps = st.session_state.get("ll_fps", 30.0)
    step = max(1, int(round(fps * interval)))

    ok, frame = cap.read()
    if not ok:  # end of stream (or a dropped connection) -> name whatever was seen
        _finish_live(st.session_state["ll_video"], threshold, cluster_sim, done=True)
        st.rerun(scope="app")
        return
    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # index of the next frame; we just read pos-1
    ts = max(pos - 1, 0) / fps
    faces = C.process_frame(get_app(), known, frame, ts, clusters, cluster_sim=cluster_sim)
    # Label each live box by its cluster's running 3-of-5 score (the same decision finalize()
    # makes on Stop), not the raw per-frame argmax, so a name stays stable instead of
    # flickering, and a single hard frame can't flash a wrong identity. Display-only: clusters,
    # the 0.40 threshold and the Stop->naming path are untouched. To revert, delete this block
    # and the `smooth.get(...)` lookup below, restoring `hit`/`label` from `f` directly.
    here = round(ts, 2)
    smooth: dict[tuple, tuple] = {}  # bbox seen this frame -> (person, score, recognized)
    for c in clusters:
        o = c["obs"][-1]
        if o["t"] != here:  # this apparent person wasn't in the current sampled frame
            continue
        cand = max(known, key=lambda p: max(ob["sims"][p] for ob in c["obs"]))
        series = [ob["sims"][cand] for ob in c["obs"]]
        tsim = C.temporal_score(series)
        recognized = tsim is not None and tsim >= threshold
        smooth[tuple(o["bbox"])] = (cand, tsim if tsim is not None else max(series), recognized)
    for f in faces:
        person, score, hit = smooth.get(
            tuple(f["bbox"]), (f["person"], f["sim"], f["sim"] >= threshold)
        )
        if not hit and not show_unknown:
            continue
        x1, y1, x2, y2 = f["bbox"]
        color = (0, 220, 0) if hit else (150, 150, 150)
        label = f"{L.short_name(person)} {score:.2f}" if hit else f"? {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        L.draw_label(frame, x1, y1, label, color)
    st.session_state["ll_last_frame"] = frame
    st.image(frame, channels="BGR", width=width)
    st.caption(f"t = {ts:5.1f}s · {len(clusters)} cluster(s)")
    for _ in range(step - 1):  # skip ahead to the next sample for the next tick
        if not cap.grab():
            break
    st.session_state["ll_pos"] = int(cap.get(cv2.CAP_PROP_POS_FRAMES))


def _live_source_picker():
    """Pick what to stream: a YouTube link resolved to a live stream (no download), or a local
    library / uploaded file. Returns (src, vid_id, label) once chosen, else None. `src` is what
    OpenCV opens; `vid_id` names the manifest folder."""
    url = st.text_input(
        "YouTube link (streamed live)", key="ll_url", placeholder="https://youtube.com/watch?v=..."
    )
    if st.button("▶ Stream & recognize", type="primary", disabled=not url, key="ll_stream_go"):
        try:
            with st.spinner("Resolving the live stream..."):
                media, vid, title = VL.stream_url(url)
        except Exception as e:
            st.error(f"Couldn't open that link as a live stream: {e}")
            return None
        return media, f"live_{vid}", title

    vids = VL.library()
    options = ["- none -"] + [str(p) for p in vids]
    lib_col, up_col = st.columns(2)
    with lib_col:
        picked = st.selectbox(
            "...or a local library file",
            options,
            key="ll_pick",
            format_func=lambda s: s if s == "- none -" else VL.title_for(s),
        )
    with up_col:
        uploaded = st.file_uploader(
            "...or upload a file", type=[e[1:] for e in VIDEO_EXTS], key="ll_up"
        )
    if uploaded is not None:
        p = save_upload(uploaded)
        return str(p), Path(p).stem, VL.title_for(p)
    if picked != "- none -":
        return picked, Path(picked).stem, VL.title_for(picked)
    return None


def page_live_learn() -> None:
    """Continuous flow: paste a link -> it streams with recognition -> Stop -> name the unknowns."""
    gallery_path, interval, threshold, cluster_sim = _settings()
    st.header("Live + learn")
    if not G.embeddings_from(gallery_path):
        st.error("The gallery is empty, enroll someone first (People ▸ Add person).")
        return

    # Review mode: the stream ended / was stopped -> name the unknowns with the shared cards.
    if st.session_state.get("ll_review"):
        st.success("End of stream." if st.session_state.get("ll_done") else "Stopped.")
        again, newsrc = st.columns(2)
        if again.button("▶ Play this stream again", type="primary"):
            _reset_live(st.session_state["ll_src"], st.session_state["ll_video"], gallery_path)
            st.session_state["ll_playing"] = True
            st.rerun()
        if newsrc.button("⟲ New source"):
            _clear_live()
            st.rerun()
        R.render_manifest(st.session_state["ll_review"], gallery_path, PHOTOS_ROOT)
        return

    # No source yet -> pick one, then start streaming immediately (the whole point).
    if not st.session_state.get("ll_src"):
        chosen = _live_source_picker()
        if chosen:
            src, vid_id, label = chosen
            _reset_live(src, vid_id, gallery_path)
            st.session_state["ll_title"] = label
            st.session_state["ll_playing"] = True
            st.rerun()
        return

    # A source is loaded -> transport controls + the live player.
    st.caption(f"**{st.session_state.get('ll_title', st.session_state['ll_video'])}**")
    playing = st.session_state.get("ll_playing", False)
    opt1, opt2 = st.columns([2, 3])
    show_unknown = opt1.checkbox(
        "Also box unrecognized faces (gray '?')", value=True, key="ll_show_unknown"
    )
    size = opt2.select_slider(
        "Video size", options=["Small", "Medium", "Large", "X-Large"], value="Medium", key="ll_size"
    )
    width = {"Small": 400, "Medium": 600, "Large": 820, "X-Large": 1040}[size]
    c1, c2, c3 = st.columns(3)
    if not playing:
        if c1.button("▶ Play", type="primary", key="ll_play"):
            st.session_state["ll_playing"] = True
            st.rerun()
    elif c1.button("⏸ Pause", key="ll_pause"):
        st.session_state["ll_playing"] = False
        st.rerun()
    if c2.button("⏹ Stop & name unknowns", key="ll_stop"):
        _finish_live(st.session_state["ll_video"], threshold, cluster_sim, done=False)
        st.rerun()
    if c3.button("⟲ New source", key="ll_new"):
        _clear_live()
        st.rerun()
    _live_fragment(threshold, cluster_sim, interval, show_unknown, width)


def page_add() -> None:
    """Add a new person from Wikimedia or from your own photos."""
    gallery_path, _i, threshold, _c = _settings()
    render_add_person(gallery_path, PHOTOS_ROOT, threshold)


def page_gallery() -> None:
    """Browse and manage the enrolled gallery."""
    gallery_path, *_ = _settings()
    render_gallery(gallery_path, PHOTOS_ROOT)


def main() -> None:
    st.set_page_config(page_title="Video face recognition", layout="wide")
    nav = st.navigation(
        {
            "Videos": [
                st.Page(page_recognize, title="Recognize", default=True),
                st.Page(page_live_learn, title="Live + learn"),
            ],
            "People": [
                st.Page(page_add, title="Add person"),
                st.Page(page_gallery, title="Gallery"),
            ],
        }
    )
    nav.run()


if __name__ == "__main__":
    main()
