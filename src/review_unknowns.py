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


def load_manifest(path: str) -> dict:
    return json.loads(Path(path).read_text())


def save_manifest(manifest: dict, path: str) -> None:
    Path(path).write_text(json.dumps(manifest, indent=1))


def log_decision(mdir: Path, record: dict) -> None:
    record = {"ts": _dt.datetime.now().isoformat(timespec="seconds"), **record}
    with open(mdir / "decisions.jsonl", "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


CONSISTENCY_MIN = 0.6  # re-detected crop face must still match the clustered face this closely


def preflight(
    crop_path: Path,
    cluster_emb: list[float],
    gallery_path: str,
    threshold: float,
    proposed_name: str,
) -> tuple[dict | None, list[str]]:
    """Embed the crop exactly as enroll() will, and return (record, warnings). enroll() picks
    the most confident face in the crop, on a crowded crop that can be a different person than
    the one we clustered (e.g. someone in the background), so we validate the face that will
    actually be enrolled, not the manifest embedding. Warnings the reviewer must override:
      * no face          -> can't enroll at all
      * multiple faces   -> ambiguous which one enroll() will pick
      * mismatch         -> the crop's main face isn't the one detected in the video
      * contamination    -> that face is >= threshold-close to a different enrolled person
    """
    rec = G.embed_photo(crop_path)  # loads the shared detector/embedder on first use
    if rec is None:
        return None, ["no face could be detected in this crop"]
    emb = np.asarray(rec["emb"], dtype=np.float32)

    warnings: list[str] = []
    if "multiple_faces" in rec["flags"]:
        warnings.append(
            "the crop contains more than one face, enroll() will pick the most "
            "confident one, which may not be this person"
        )
    consistency = float(np.dot(G.l2(emb), G.l2(np.asarray(cluster_emb, dtype=np.float32))))
    if consistency < CONSISTENCY_MIN:
        warnings.append(
            f"the crop's main face differs from the one seen in the video "
            f"(cosine {consistency:.2f}), it may be a nearby/background person"
        )
    if Path(gallery_path).exists():
        known = G.load_embeddings(gallery_path)
        if known:
            other, sim = G.match(emb, known)
            if sim >= threshold and other != proposed_name:
                warnings.append(
                    f"this face is {sim:.2f}-close to **{other}**, already enrolled, "
                    f"likely a mislabel"
                )
    return rec, warnings


def enroll_crop(crop_path: Path, name: str, gallery_path: str, photos_root: str) -> int:
    """File the crop under data/reference_photos/<name>/ and enroll it through the single
    gallery.enroll() path. Returns embeddings added (0 if no face re-detected in the crop).
    Call `preflight` first, this assumes the crop has already been validated."""
    dest_dir = Path(photos_root) / name
    dest_dir.mkdir(parents=True, exist_ok=True)
    n_existing = len([p for p in dest_dir.iterdir() if p.suffix.lower() in G.IMAGE_EXTS])
    dest = dest_dir / f"{name}_{n_existing}{crop_path.suffix.lower()}"
    shutil.copy(crop_path, dest)

    gallery = G.load(gallery_path) if Path(gallery_path).exists() else G.new_gallery()
    added = G.enroll(gallery, name, [dest], verbose=False)
    if added == 0:
        dest.unlink(missing_ok=True)  # no face re-detected in the crop, don't leave a dead file
        return 0
    G.touch(gallery)
    G.save(gallery, gallery_path)
    return added


def short_name(person: str) -> str:
    return person.split("_")[-1].capitalize()  # ana_maria_silva -> "Silva"


def pretty(person: str) -> str:
    return person.replace("_", " ").title()  # ana_maria_silva -> "Ana Maria Silva"


NEW_PERSON = "New person..."


def gallery_people(gallery_path: str) -> list[str]:
    if not Path(gallery_path).exists():
        return []
    return sorted(G.load(gallery_path)["people"].keys())


def name_picker(cid, people: list[str], default: str | None = None):
    """A 'who is this?' control: pick an existing gallery person, or add a new one. Returns
    the chosen person key (snake_case), or '' if 'new person' is chosen with an empty box.
    Picking an existing person is how a missed face (e.g. a frame of someone already enrolled) gets
    added to that person's gallery entry instead of created as a duplicate identity."""
    options = [NEW_PERSON] + people
    index = options.index(default) if default in options else 0
    choice = st.selectbox(
        "Who is this?",
        options,
        index=index,
        key=f"pick_{cid}",
        format_func=lambda x: x if x == NEW_PERSON else pretty(x),
    )
    if choice == NEW_PERSON:
        raw = st.text_input("New person's name", key=f"new_{cid}", placeholder="e.g. İlkay Işık")
        return norm_name(raw)
    return choice


def do_enroll(
    c, manifest, manifest_path, gallery_path, photos_root, threshold, name, override, action
) -> None:
    """Shared enroll/relabel action: run the preflight guard, enroll the crop through the
    single gallery.enroll() path, update the manifest + audit log, then rerun. `action` is
    'enroll' (an unknown face) or 'relabel' (correcting/reinforcing a recognized one)."""
    cid = c["id"]
    mdir = Path(manifest_path).parent  # crops + decisions.jsonl sit beside the manifest
    # Never st.stop() here, do_enroll runs mid-render inside a card, and st.stop()
    # kills the rest of the page (every card below this one). Plain returns keep the page.
    if not name:
        st.warning("Pick or type a name first.")
        return
    rec, warnings = preflight(mdir / c["crop"], c["emb"], gallery_path, threshold, name)
    if rec is None:
        st.error(
            "No face re-detected in this crop, even at a lowered detection threshold, "
            "too blurred / extreme angle to use as a reference. Not enrolled."
        )
        return
    if warnings and not override:
        for w in warnings:
            st.warning(w)
        st.info("If you're sure, tick the override box and press the button again.")
        return
    added = enroll_crop(mdir / c["crop"], name, gallery_path, photos_root)
    if added == 0:
        st.error("No face re-detected in this crop, not enrolled.")
        return
    c["status"] = f"{action}:{name}"
    save_manifest(manifest, manifest_path)
    log_decision(
        mdir,
        {
            "cluster": cid,
            "action": action,
            "name": name,
            "override": bool(warnings and override),
            "warnings": warnings,
            "was": c.get("match"),
            "sim": c.get("sim"),
        },
    )
    st.success(f"Saved as **{pretty(name)}** (+{added} embedding). Gallery updated.")
    st.rerun()


# ---------------------------------------------------------------------------
# UI  (render_manifest, the review cards, rendered inside the app's pages)
# ---------------------------------------------------------------------------
def render_manifest(manifest_path: str, gallery_path: str, photos_root: str) -> None:
    """Two sections: (1) unknown faces to identify, name (from the gallery or new) or ignore;
    (2) recognized faces, shown with their match so the user can correct a wrong one or
    reinforce a right one. Both go through the same guarded enroll path. Reused by the
    standalone review script and the upload-driven app."""
    mpath = Path(manifest_path)
    mdir = mpath.parent
    manifest = load_manifest(manifest_path)
    threshold = manifest.get("threshold", G.DEFAULT_THRESHOLD)
    people = gallery_people(gallery_path)
    clusters = manifest["clusters"]

    to_identify = [c for c in clusters if not c.get("recognized") and c["status"] == "pending"]
    recognized = [c for c in clusters if c.get("recognized")]
    resolved = [c for c in clusters if not c.get("recognized") and c["status"] != "pending"]

    # --- Section 1: unknown faces ------------------------------------------
    st.subheader(f"Unknown faces ({len(to_identify)})")
    if not to_identify:
        st.success("No unknown faces.")
    for c in to_identify:
        cid = c["id"]
        st.divider()
        col_img, col_form = st.columns([1, 2])
        with col_img:
            crop = mdir / c["crop"]
            if crop.exists():
                st.image(str(crop), width=200)
        with col_form:
            span = f"{c['n_frames']} frames · {c['first_t']:.1f}-{c['last_t']:.1f}s"
            near = f"nearest {short_name(c['match'])} {c['sim']:.2f}"
            tail = (
                "<3 samples"
                if c["temporal_sim"] is None
                else f"{manifest['rule']} {c['temporal_sim']:.2f} < {threshold}"
            )
            st.caption(f"{span} · {near} · {tail}")
            name = name_picker(cid, people)
            override = st.checkbox(
                "Enroll anyway (override the warnings below)", key=f"override_{cid}"
            )
            b_enroll, b_skip, _ = st.columns([1, 1, 3])
            if b_enroll.button("Enroll", key=f"enroll_{cid}", type="primary"):
                do_enroll(
                    c,
                    manifest,
                    manifest_path,
                    gallery_path,
                    photos_root,
                    threshold,
                    name,
                    override,
                    "enroll",
                )
            if b_skip.button("Ignore", key=f"skip_{cid}"):
                c["status"] = "skipped"
                save_manifest(manifest, manifest_path)
                C.add_ignored(mdir, c["emb"])  # remember, so re-scans don't re-ask this face
                log_decision(mdir, {"cluster": cid, "action": "skip"})
                st.rerun()

    # --- Section 2: recognized faces (confirm / correct) -------------------
    if recognized:
        st.subheader(f"Recognized ({len(recognized)})")
        for c in recognized:
            cid = c["id"]
            st.divider()
            relabeled = c["status"].startswith("relabel:")
            current = c["status"].split(":", 1)[1] if relabeled else c["match"]
            col_img, col_form = st.columns([1, 2])
            with col_img:
                crop = mdir / c["crop"]
                if crop.exists():
                    st.image(str(crop), width=200)
            with col_form:
                tag = " · corrected" if relabeled else ""
                score = f"peak {c['sim']:.2f} · {manifest['rule']} {c['temporal_sim']:.2f}"
                st.markdown(f"**{pretty(current)}** · {score}{tag}")
                st.caption(f"{c['n_frames']} frames · {c['first_t']:.1f}-{c['last_t']:.1f}s")
                with st.expander("Fix / add to gallery"):
                    name = name_picker(cid, people, default=current if current in people else None)
                    override = st.checkbox(
                        "Save anyway (override the warnings below)", key=f"override_{cid}"
                    )
                    if st.button("Save to gallery", key=f"relabel_{cid}"):
                        do_enroll(
                            c,
                            manifest,
                            manifest_path,
                            gallery_path,
                            photos_root,
                            threshold,
                            name,
                            override,
                            "relabel",
                        )

    # --- handled unknowns --------------------------------------------------
    if resolved:
        with st.expander(f"Handled unknowns ({len(resolved)})"):
            for c in resolved:
                st.write(
                    f"Face {c['id']}, {c['status']}  "
                    f"({c['n_frames']} frames, {c['first_t']:.1f}s-{c['last_t']:.1f}s)"
                )

    n_ignored = len(C.load_ignored(mdir))
    if n_ignored:
        with st.expander(f"Ignored earlier: {n_ignored} face(s), auto-skipped on every scan"):
            if st.button("Ask about them again (forget the ignore list)"):
                C.clear_ignored(mdir)
                st.rerun()
