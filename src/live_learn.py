"""
The app's "Live + learn" page, a continuous flow: paste a link -> it streams and plays with
recognition boxes (nothing is downloaded) -> press Stop -> it asks who every unrecognized face
is. (Prototype.)

No download: `video_library.stream_url` resolves the link to a direct media URL via yt-dlp
(skip_download) and OpenCV reads frames straight off it. Streamlit can't interrupt a blocking
while-loop with a button, so playback is a self-rerunning `st.fragment`: each tick runs the
shared per-frame path (`collect_unknowns.process_frame`, the same detect/match/cluster the
batch scan uses) on one sampled frame, draws boxes, and accumulates clusters in session_state.
Stop / end-of-stream runs `collect_unknowns.finalize` + `write_manifest` on what was seen, then
the existing review cards (`review_unknowns.render_manifest`) name the unknowns.

All state lives under `ll_*` session keys: `ll_src` is what OpenCV opens (stream URL or local
path), `ll_video` a short id for the manifest folder. The capture stays open across ticks and
pauses (streaming never needs to seek), and is released only on Stop / Restart / New source.

Library module: `render()` is the entry point, called by the page function in `app.py`.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import streamlit as st

import collect_unknowns as C
import gallery as G
import live_recognition as L  # per-frame box/label drawing, shared with the cv2 window
import review_unknowns as R
import video_library as VL

WIDTHS = {"Small": 400, "Medium": 600, "Large": 820, "X-Large": 1040}  # on-screen player sizes


def _release_capture() -> None:
    cap = st.session_state.pop("ll_cap", None)
    if cap is not None:
        cap.release()


def _reset(src, vid_id, gallery_path) -> None:
    """(Re)initialise the live session for a source: reload the gallery, drop the frame position
    / accumulated clusters / any capture, and leave it paused at the start of `src`."""
    _release_capture()
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


def _clear() -> None:
    """Tear the live session down entirely (back to the source picker), everything `_reset`
    sets, plus the title/fps the source picker and the capture add."""
    _release_capture()
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


def _capture(src):
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


def _finish(vid_id, threshold, cluster_sim, *, done: bool) -> None:
    """Turn what the live pass has seen so far into a review manifest: finalize the clusters
    with the 3-of-5 rule, write crops + manifest.json under results/unknowns/<vid_id>/, then
    flip into review mode. `done` distinguishes reaching the stream's end from a manual Stop."""
    clusters = st.session_state.get("ll_clusters", [])
    known = st.session_state.get("ll_known") or {}
    C.finalize(clusters, known, threshold)
    mpath = C.write_manifest(clusters, vid_id, threshold=threshold, cluster_sim=cluster_sim)
    _release_capture()
    st.session_state.update(ll_playing=False, ll_review=str(mpath), ll_done=done)


@st.fragment(run_every="0.15s")
def _fragment(threshold, cluster_sim, interval, show_unknown, width):
    """One playback tick, auto-rerun on a timer (so a Stop/Pause button, rendered outside the
    fragment, can still be clicked between ticks). Reads the next sampled frame off the open
    stream, accumulates clusters via the shared path, draws boxes, and advances. On end-of-
    stream it finalises and reruns the whole app into review mode. `width` caps the on-screen
    frame size (px) so the player isn't full-page-wide.

    One frame per tick: an in-tick playback loop fights the fragment timer (reruns overlap
    the loop and playback stutters/jumps). Smooth playback
    is the standalone window's job (live_recognition.py)."""
    if not st.session_state.get("ll_playing"):
        last = st.session_state.get("ll_last_frame")
        if last is not None:  # keep the paused frame on screen instead of a blank gap
            st.image(last, channels="BGR", width=width)
            st.caption("⏸ paused")
        return
    known = st.session_state.get("ll_known") or {}
    clusters = st.session_state.setdefault("ll_clusters", [])
    cap = _capture(st.session_state["ll_src"])
    fps = st.session_state.get("ll_fps", 30.0)
    step = max(1, int(round(fps * interval)))

    ok, frame = cap.read()
    if not ok:  # end of stream (or a dropped connection) -> name whatever was seen
        _finish(st.session_state["ll_video"], threshold, cluster_sim, done=True)
        st.rerun(scope="app")
        return
    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # index of the next frame; we just read pos-1
    ts = max(pos - 1, 0) / fps
    faces = C.process_frame(G.get_app(), known, frame, ts, clusters, cluster_sim=cluster_sim)
    # Label each live box by its cluster's running 3-of-5 verdict (C.decide_cluster, the same
    # decision finalize() makes on Stop), not the raw per-frame argmax, so a name stays stable
    # instead of flickering, and a single hard frame can't flash a wrong identity. Display-only:
    # clusters, the 0.40 threshold and the Stop->naming path are untouched.
    here = round(ts, 2)
    smooth: dict[tuple, tuple] = {}  # bbox seen this frame -> (person, score, recognized)
    for c in clusters:
        o = c["obs"][-1]
        if o["t"] != here:  # this apparent person wasn't in the current sampled frame
            continue
        d = C.decide_cluster(c["obs"], known, threshold)
        # Too brief for the rule yet -> show the peak so the box isn't scoreless.
        score = d["peak_sim"] if d["temporal_sim"] is None else d["temporal_sim"]
        smooth[tuple(o["bbox"])] = (d["candidate"], score, d["recognized"])
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


def _source_picker():
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
            "...or upload a file", type=[e[1:] for e in VL.VIDEO_EXTS], key="ll_up"
        )
    if uploaded is not None:
        p = VL.save_upload(uploaded.name, uploaded.getbuffer())
        return str(p), Path(p).stem, VL.title_for(p)
    if picked != "- none -":
        return picked, Path(picked).stem, VL.title_for(picked)
    return None


def render(gallery_path, *, interval, threshold, cluster_sim) -> None:
    """The page body: paste a link -> it streams with recognition -> Stop -> name the unknowns.
    The four settings come from the app's sidebar (see `app._settings`)."""
    st.header("Live + learn")
    if not G.embeddings_from(gallery_path):
        st.error("The gallery is empty, enroll someone first (People ▸ Add person).")
        return

    # Review mode: the stream ended / was stopped -> name the unknowns with the shared cards.
    if st.session_state.get("ll_review"):
        st.success("End of stream." if st.session_state.get("ll_done") else "Stopped.")
        again, newsrc = st.columns(2)
        if again.button("▶ Play this stream again", type="primary"):
            _reset(st.session_state["ll_src"], st.session_state["ll_video"], gallery_path)
            st.session_state["ll_playing"] = True
            st.rerun()
        if newsrc.button("⟲ New source"):
            _clear()
            st.rerun()
        R.render_manifest(st.session_state["ll_review"], gallery_path, G.DEFAULT_PHOTOS)
        return

    # No source yet -> pick one, then start streaming immediately (the whole point).
    if not st.session_state.get("ll_src"):
        chosen = _source_picker()
        if chosen:
            src, vid_id, label = chosen
            _reset(src, vid_id, gallery_path)
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
    size = opt2.select_slider("Video size", options=list(WIDTHS), value="Medium", key="ll_size")
    c1, c2, c3 = st.columns(3)
    if not playing:
        if c1.button("▶ Play", type="primary", key="ll_play"):
            st.session_state["ll_playing"] = True
            st.rerun()
    elif c1.button("⏸ Pause", key="ll_pause"):
        st.session_state["ll_playing"] = False
        st.rerun()
    if c2.button("⏹ Stop & name unknowns", key="ll_stop"):
        _finish(st.session_state["ll_video"], threshold, cluster_sim, done=False)
        st.rerun()
    if c3.button("⟲ New source", key="ll_new"):
        _clear()
        st.rerun()
    _fragment(threshold, cluster_sim, interval, show_unknown, WIDTHS[size])
