#!/usr/bin/env python3
"""
Watch recognition happen live in a window, the real-time overlay.

Full detection + match runs on every frame: each face is embedded and compared to the enrolled
gallery, and only the ones it recognizes (cosine >= the calibrated 0.40 threshold) get a green
box + name. Everyone it doesn't recognize is left alone, no box, no label, no learning (use
--show-unknown to also draw sub-threshold faces as a gray "?"). The label reacts instantly, and
boxes always sit on the actual detected face.

Source can be a video file, a webcam (--video 0), or a YouTube link (streamed live via
video_library.stream_url, nothing is downloaded). Runs on the CoreML provider (~42 ms/frame on
M1 Pro, see gallery.get_app).

>>> This opens a GUI window, so run it yourself in your terminal, the window shows on your
    screen (a background/automation process can't display it). Controls: 'q' or ESC = quit,
    space = pause/resume.

Run it with no arguments and it becomes an interactive tool: it asks for a source, plays it to
the end, then asks whether you want another one. The model and the gallery are loaded once and
reused for every video in the session (the load is the slow part, a few seconds).

Usage:
    python src/live_recognition.py                                  # interactive: asks each time
    python src/live_recognition.py --video data/videos/video_01.mp4
    python src/live_recognition.py --video 0                        # webcam
    python src/live_recognition.py --video "https://youtu.be/XXXX"  # YouTube (streamed, no download)
    python src/live_recognition.py --video X.mp4 --show-unknown --record results/live.mp4
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np

import gallery as G  # shared CoreML detector/embedder + gallery match
import video_library as VL  # resolve a YouTube link to a stream URL (no download)

MIN_DET_SCORE = 0.5


def short_name(person: str) -> str:
    return person.split("_")[-1].capitalize()  # ana_maria_silva -> "Silva"


def draw_label(frame, x1, y1, text, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    ytxt = max(th + 6, y1 - 6)
    xtxt = min(x1, frame.shape[1] - tw - 4)  # keep label on-screen
    cv2.rectangle(frame, (xtxt, ytxt - th - 6), (xtxt + tw + 4, ytxt + 2), color, -1)
    cv2.putText(frame, text, (xtxt + 2, ytxt - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def annotate(frame, app, gallery, threshold, show_unknown) -> int:
    """Draw boxes/labels on `frame` in place; return the number of recognized (>=thr) faces."""
    hits = 0
    for d in app.get(frame):
        if d.det_score < MIN_DET_SCORE:
            continue
        x1, y1, x2, y2 = map(int, d.bbox)
        emb = np.asarray(d.normed_embedding, dtype=np.float32)
        person, sim = G.match(emb, gallery)
        hit = sim >= threshold
        if not hit and not show_unknown:
            continue  # sub-threshold -> hidden (same as the annotated .mp4)
        color = (0, 220, 0) if hit else (150, 150, 150)
        label = f"{short_name(person)} {sim:.2f}" if hit else f"? {sim:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_label(frame, x1, y1, label, color)
        hits += hit
    return hits


def open_capture(spec: str):
    """Open one source (file path, camera index ('0'), or link) as a cv2.VideoCapture.

    Raises ValueError if it can't be opened, so the interactive loop can just ask again.
    """
    if spec.isdigit():
        src: int | str = int(spec)  # webcam index
    elif re.match(r"^(https?://|www\.)", spec):
        # A YouTube (or other) link: resolve it to a directly-readable media URL without
        # downloading anything (same path the app's Live+learn page uses), then let OpenCV
        # read frames straight off it.
        print(f"Resolving stream for {spec} ...")
        src, _vid, title = VL.stream_url(spec)
        print(f"Streaming: {title}")
    else:
        src = str(Path(spec).expanduser())  # local file path
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not open video source: {spec!r}")
    return cap


def watch(cap, app, gallery, *, threshold, show_unknown, record, headless, limit) -> None:
    """Play one source to the end (or until 'q'), drawing recognitions on every frame."""
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if record:
        Path(record).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(record, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

    win = "recognition  (q/ESC = quit, space = pause)"
    if not headless:
        print("Live window opening, focus it, then press 'q' or ESC to quit, space to pause.")

    idx = 0
    fps = 0.0
    t_last = time.perf_counter()
    paused = False
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            annotate(frame, app, gallery, threshold, show_unknown)
            now = time.perf_counter()
            dt = now - t_last
            t_last = now
            fps = (0.9 * fps + 0.1 / dt) if dt > 0 else fps  # smoothed processing fps
            cv2.putText(
                frame,
                f"{fps:4.1f} fps",
                (8, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            if writer is not None:
                writer.write(frame)
            idx += 1

        if not headless:
            cv2.imshow(win, frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord(" "):
                paused = not paused
        if limit and idx >= limit:
            break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Recorded -> {record}")
    if not headless:
        cv2.destroyAllWindows()
    print(f"Done, processed {idx} frames (~{fps:.1f} fps).")


def ask(text: str) -> str:
    """One prompt. Ctrl-C / Ctrl-D read as 'quit' rather than a traceback."""
    try:
        return input(text).strip().strip("'\"")  # drag-dropped paths arrive quoted
    except (EOFError, KeyboardInterrupt):
        print()
        return "q"


def ask_source() -> str | None:
    """Ask for one video source; returns a spec for open_capture(), or None to quit."""
    while True:
        print(
            "\nSource?\n  1) YouTube link (streamed, nothing downloaded)\n  2) Video file"
            "\n  3) Webcam\n  q) Quit"
        )
        choice = ask("> ").lower()
        if choice in ("q", "quit", "4"):
            return None
        if choice == "1":
            if url := ask("Link: "):
                return url
        elif choice == "2":
            path = ask("Path: ")
            if not path:
                continue
            if not Path(path).expanduser().exists():
                print(f"No such file: {path}")
                continue
            return path
        elif choice == "3":
            return ask("Camera index [0]: ") or "0"
        else:
            print("Pick 1, 2, 3 or q.")


def interactive(app, gallery, args) -> None:
    """Ask -> play -> ask again, reusing the already-loaded model and gallery."""
    while True:
        spec = ask_source()
        if spec is None:
            break
        record = ask("Save annotated mp4? path (blank = no): ") or None
        if record in ("q", "quit"):  # a stray quit at the record prompt shouldn't start a video
            break
        try:
            cap = open_capture(spec)
        except ValueError as e:
            print(e)
            continue
        watch(
            cap,
            app,
            gallery,
            threshold=args.threshold,
            show_unknown=args.show_unknown,
            record=record,
            headless=False,
            limit=args.limit,
        )
        if ask("\nAnother video? [Y/n] ").lower() in ("n", "no", "q", "quit"):
            break
    print("Bye.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--video",
        help="a video file path, a camera index like 0 for webcam, or a YouTube link "
        "(streamed live, nothing is downloaded). Omit it to be asked interactively.",
    )
    ap.add_argument("--gallery", default="data/gallery.json")
    ap.add_argument(
        "--threshold", type=float, default=G.DEFAULT_THRESHOLD, help="match cutoff (calibrated)"
    )
    ap.add_argument(
        "--show-unknown", action="store_true", help="also draw sub-threshold faces as a gray '?'"
    )
    ap.add_argument("--record", default=None, help="also save the annotated stream to this mp4")
    ap.add_argument(
        "--headless",
        action="store_true",
        help="no GUI window (for testing), process --limit frames and exit",
    )
    ap.add_argument("--limit", type=int, default=0, help="stop after N frames (0 = all)")
    args = ap.parse_args()

    app = G.get_app()
    print(f"Loading gallery from {args.gallery} ...")
    gallery = G.embeddings_from(args.gallery)

    if not args.video:
        if args.headless:
            raise SystemExit("--headless needs --video (there is nobody to ask).")
        interactive(app, gallery, args)
        return

    try:
        cap = open_capture(args.video)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    watch(
        cap,
        app,
        gallery,
        threshold=args.threshold,
        show_unknown=args.show_unknown,
        record=args.record,
        headless=args.headless,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
