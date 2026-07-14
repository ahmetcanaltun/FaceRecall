"""
A tidy video library for the app. Files named after raw YouTube IDs
(`clip_xOWC1zvkZ3E.mp4`) are meaningless to a human. Instead:

  * on disk, files get **simple sequential names**, `video_01.mp4`, `video_02.mp4`, ...,
    so the folder stays clean and ordered;
  * after a scan, a video can be **renamed to the people in it**, e.g. `silva_petersen.mp4`;
  * the real title / source URL / recognized people live in an index (`data/videos/library.json`,
    keyed by YouTube id) so the app can still show a meaningful label in the picker.

Downloads fetch the best video up to 1080p merged with audio when **ffmpeg** is available
(much sharper than a single progressive stream, which YouTube caps low); without ffmpeg they
fall back to the best progressive mp4. An optional **start/end time range** downloads only that
section (needs ffmpeg). Re-adding a full URL already in the library is a no-op (dedup by id).

Library module: used by the app (the "Download from YouTube" box, the picker, the
rename-to-people button, and the Live+learn page's `stream_url`).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)

VIDEOS_DIR = "data/videos"
INDEX_NAME = "library.json"
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")


# ---------------------------------------------------------------------------
# index  (keyed by youtube id: {id: {file, title, url, added, people:[...]}})
# ---------------------------------------------------------------------------
def _index_path(dest_dir: str = VIDEOS_DIR) -> Path:
    return Path(dest_dir) / INDEX_NAME


def load_index(dest_dir: str = VIDEOS_DIR) -> dict:
    p = _index_path(dest_dir)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save_index(idx: dict, dest_dir: str = VIDEOS_DIR) -> None:
    _index_path(dest_dir).write_text(
        json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def _index_put(
    vid: str,
    *,
    file: str,
    title: str,
    url: str,
    people: list[str] | None = None,
    dest_dir: str = VIDEOS_DIR,
) -> None:
    idx = load_index(dest_dir)
    entry = idx.get(vid, {})
    entry.update(
        file=file, title=title, url=url, added=entry.get("added", date.today().isoformat())
    )
    if people is not None:
        entry["people"] = people
    idx[vid] = entry
    _save_index(idx, dest_dir)


def _index_entry_for_file(fname: str, dest_dir: str = VIDEOS_DIR):
    for vid, meta in load_index(dest_dir).items():
        if meta.get("file") == fname:
            return vid, meta
    return None, None


# ---------------------------------------------------------------------------
# naming
# ---------------------------------------------------------------------------
def _next_number(dest_dir: str) -> int:
    nums = [
        int(m.group(1))
        for p in Path(dest_dir).glob("video_*")
        if (m := re.match(r"video_(\d+)", p.stem))
    ]
    return (max(nums) + 1) if nums else 1


def sequential_name(dest_dir: str, ext: str = "mp4") -> str:
    return f"video_{_next_number(dest_dir):02d}.{ext}"


def people_name(people: list[str], ext: str = "mp4") -> str:
    """A filename from the people in a clip: ana_maria_silva + lars_petersen -> silva_petersen."""
    shorts = [p.split("_")[-1] for p in people][:3]
    base = "_".join(shorts) if shorts else "unknown"
    if len(people) > 3:
        base += f"_+{len(people) - 3}"
    return f"{base}.{ext}"


# ---------------------------------------------------------------------------
# listing (for the app's picker)
# ---------------------------------------------------------------------------
def library(dest_dir: str = VIDEOS_DIR) -> list[Path]:
    root = Path(dest_dir)
    if not root.exists():
        return []
    return sorted(
        (p for p in root.iterdir() if p.suffix.lower() in VIDEO_EXTS), key=lambda p: p.name.lower()
    )


def title_for(path, dest_dir: str = VIDEOS_DIR) -> str:
    """A meaningful label for a video: the people in it if known, else the source title,
    else the bare filename."""
    _, meta = _index_entry_for_file(Path(path).name, dest_dir)
    if meta:
        if meta.get("people"):
            return ", ".join(p.split("_")[-1].title() for p in meta["people"])
        if meta.get("title"):
            return meta["title"]
    return Path(path).stem


# ---------------------------------------------------------------------------
# operations
# ---------------------------------------------------------------------------
def has_ffmpeg() -> bool:
    """ffmpeg is needed to merge the best video+audio streams and to cut a time range."""
    import shutil

    return shutil.which("ffmpeg") is not None


def _hms(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 60}:{sec % 60:02d}"


def download(
    url: str,
    dest_dir: str = VIDEOS_DIR,
    progress_hook=None,
    start: float | None = None,
    end: float | None = None,
) -> Path:
    """Download one clip, name it `video_NN`, index it, return its path.

    Quality: with ffmpeg, grabs the best video up to 1080p merged with audio (YouTube serves
    high resolutions only as separate streams, so a single progressive file is capped low,
    this is the fix for "downloads are always low quality"). Without ffmpeg, falls back to the
    best progressive mp4.

    Section: if `start`/`end` (seconds) are given, downloads only that time range, this needs
    ffmpeg. A full re-download of a URL already in the library is skipped (dedup by id); a
    section request always downloads (you may want a different span)."""
    import yt_dlp

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    ffmpeg = has_ffmpeg()
    section = start is not None or end is not None
    if section and not ffmpeg:
        raise RuntimeError(
            "Downloading a time range needs ffmpeg, install it "
            "(e.g. `brew install ffmpeg`), then try again."
        )

    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    vid = info["id"]
    idx = load_index(dest_dir)
    if not section and vid in idx and (dest / idx[vid]["file"]).exists():
        return dest / idx[vid]["file"]  # already have the full clip

    fname = sequential_name(dest_dir)
    # Prefer mp4 video + m4a audio so the merge into an .mp4 container is a clean remux, mixing
    # a webm/VP9 video or opus audio into mp4 is what makes ffmpeg fail (the "code 222" merge
    # error). Fall back to any best streams, then to a single progressive file.
    fmt = (
        "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/"
        "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best"
        if ffmpeg
        else "best[ext=mp4][height<=1080]/best[ext=mp4]/best"
    )
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "format": fmt,
        "outtmpl": str(dest / fname),
    }
    if ffmpeg:
        opts["merge_output_format"] = "mp4"
    if section:
        from yt_dlp.utils import download_range_func

        s = float(start or 0)
        e = float(end) if end is not None else float(info.get("duration") or s + 3600)
        opts["download_ranges"] = download_range_func(None, [(s, e)])
        # Cut at the nearest keyframes with a stream copy (no re-encode), robust and fast;
        # the clip may start a second or two off the exact time, which is fine for scanning.
        # (force_keyframes_at_cuts re-encodes for exact cuts and is what tends to fail.)
    if progress_hook:
        opts["progress_hooks"] = [progress_hook]

    def _run(o):
        with yt_dlp.YoutubeDL(o) as ydl:
            ydl.download([url])

    try:
        _run(opts)
    except Exception:
        # A high-quality merge can still fail on unusual codec combinations; fall back to a
        # single progressive stream (the section cut, if requested, still applies).
        log.warning(
            "high-quality merge failed for %s, falling back to a progressive stream",
            url,
            exc_info=True,
        )
        opts["format"] = "best[ext=mp4][height<=1080]/best[ext=mp4]/best"
        opts.pop("merge_output_format", None)
        _run(opts)

    title = info.get("title", fname)
    if section:
        span = f"{_hms(start or 0)}-{_hms(end)}" if end is not None else f"from {_hms(start or 0)}"
        title = f"{title} [{span}]"
    _index_put(vid, file=fname, title=title, url=url, dest_dir=dest_dir)
    return dest / fname


def stream_url(url: str) -> tuple[str, str, str]:
    """Resolve a YouTube (or other) link to a directly-readable media URL **without downloading
    anything**, for streaming straight into OpenCV (the Live+learn page). Returns
    (media_url, video_id, title).

    Prefers a *video-only* mp4/avc1 stream up to 720p: recognition needs no audio, and
    YouTube's muxed *progressive* streams cap low (~360p, itag 18), so a video-only stream
    gives a much larger face (better embeddings) at the same reach. avc1/mp4 is what
    OpenCV/ffmpeg opens most reliably over HTTP; if none is readable we fall back to the best
    muxed progressive stream (<=720p), then to yt-dlp's top-level url. Higher resolution costs
    a little more per-frame decode, but the live view is already sparse-sampled. The signed URL
    expires after a while (fine for one session)."""
    import yt_dlp

    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    formats = info.get("formats", [])

    def _pick_best(cands: list[dict]) -> dict | None:
        """Highest stream <=720p; if none is <=720p, the smallest above it (closest to cap)."""
        if not cands:
            return None
        capped = [f for f in cands if (f.get("height") or 0) <= 720]
        if capped:
            return max(capped, key=lambda f: f.get("height") or 0)
        return min(cands, key=lambda f: f.get("height") or 0)

    def _has_video(f: dict) -> bool:
        return bool(f.get("url")) and f.get("vcodec") not in (None, "none")

    # 1) video-only mp4/avc1 (no audio track), the highest-resolution option OpenCV reads well
    vonly = [
        f
        for f in formats
        if _has_video(f)
        and f.get("acodec") in (None, "none")
        and (f.get("ext") == "mp4" or (f.get("vcodec") or "").startswith("avc1"))
    ]
    # 2) fall back to a muxed progressive stream (best <=720p), then to the top-level url
    prog = [f for f in formats if _has_video(f) and f.get("acodec") not in (None, "none")]
    pick = _pick_best(vonly) or _pick_best(prog)
    media = pick["url"] if pick else info.get("url")
    if not media:
        raise RuntimeError("no readable stream URL found for this link")
    return media, info["id"], info.get("title", info["id"])


def rename_to_people(path, people: list[str], dest_dir: str = VIDEOS_DIR) -> Path:
    """Rename a video file to the people appearing in it (and record them in the index)."""
    path = Path(path)
    if not people:
        return path
    name = people_name(people, path.suffix.lstrip("."))
    dst = path.with_name(name)
    i = 2
    while dst.exists() and dst != path:
        dst = path.with_name(f"{Path(name).stem}_{i}{path.suffix}")
        i += 1
    if dst != path:
        path.rename(dst)
    vid, meta = _index_entry_for_file(path.name, dest_dir)
    if vid:
        _index_put(
            vid,
            file=dst.name,
            title=meta.get("title", dst.name),
            url=meta.get("url", ""),
            people=people,
            dest_dir=dest_dir,
        )
    return dst


def _fetch_title(vid: str) -> str | None:
    try:
        import yt_dlp

        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
            return ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False).get(
                "title"
            )
    except Exception:
        log.warning("could not fetch the title for video id %s", vid, exc_info=True)
        return None


def reorganize(dest_dir: str = VIDEOS_DIR) -> list[tuple[str, str]]:
    """Rename already-downloaded files that aren't indexed yet to `video_NN`, fetching each
    one's title (when the filename carries a recoverable YouTube id) into the index. Returns
    the (old_name, new_name) renames performed."""
    root = Path(dest_dir)
    indexed = {m["file"] for m in load_index(dest_dir).values()}
    renames: list[tuple[str, str]] = []
    for p in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if p.suffix.lower() not in VIDEO_EXTS or p.name in indexed or p.stem.startswith("video_"):
            continue
        m = _ID_RE.search(p.stem.split("_")[-1]) or _ID_RE.search(p.stem)
        if not m:
            continue  # no recoverable id, leave it alone
        vid = m.group(0)
        title = _fetch_title(vid) or p.stem
        new_name = sequential_name(dest_dir, p.suffix.lstrip("."))
        new_path = root / new_name
        if new_path != p and not new_path.exists():
            p.rename(new_path)
            renames.append((p.name, new_name))
        _index_put(
            vid, file=new_name, title=title, url=f"https://youtu.be/{vid}", dest_dir=dest_dir
        )
    return renames
