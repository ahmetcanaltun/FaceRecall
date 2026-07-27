# Face Recognition in Video

Recognize known people in real, low-quality video from **very few reference photos** (often a
single one), and **learn new people on the fly**: when the system sees a face it doesn't know,
it asks who it is, enrolls the answer, and knows them on the very next pass.

Built as a research project. The findings (backend comparison, threshold calibration,
held-out validation, the temporal decision rule) are the primary deliverable; the Streamlit
app is the working prototype that ties them together.

---

## Why embeddings, not a classifier

Three constraints rule out the obvious approach, and they are what make this project interesting.

**Open-set.** The system will meet faces belonging to nobody in the database. A classifier
trained on N people always answers "one of these N"; it cannot say *nobody*. The output has to
be a distance plus a cutoff, not a class label.

**Few-shot.** Most people here have 1-3 reference photos. You do not train a classifier on one
example per class.

**Learn without retraining.** The core requirement is: meet a stranger, ask who they are, know
them on the next pass. If that meant retraining a network, the loop would take hours instead of
a second.

So: **don't classify; embed and compare.** A pretrained network maps any face to a point in
512-dimensional space, arranged so the same person's photos land close together and different
people land far apart. Adding a person is appending a point to a JSON file. "Is this anybody I
know?" is a distance query.

The network never learns anything about our specific people. It learned, once, on ~600k
identities, a general rule for *where faces go*.

---

## How it works

```
video ──▶ SCRFD detector ──▶ 5-point align ──▶ ArcFace embedder ──▶ L2 normalise
          bbox + keypoints   → 112×112         ResNet-50, 512-d     ‖v‖ = 1
                                                                        │
                            cosine similarity vs. the enrolled gallery ─┘   [data/gallery.json]
                                                │
                    greedy clustering of one apparent person across frames
                                                │
                      3-of-5 temporal rule at the calibrated 0.40 threshold
                                                │
                   "recognized: silva (0.71)"  /  "unknown: who is this?"
```

**Detection (SCRFD, `det_10g.onnx`).** Anchor points on feature maps at strides 8/16/32 each
predict a face score, four distances to the box edges, and five keypoints; NMS removes
duplicates. Its design idea is *where to spend compute*, biased toward small faces, which news
footage is full of. Note it is SCRFD, not RetinaFace; the `buffalo_l` pack ships the former.

**Alignment.** A similarity transform (rotation + uniform scale + translation, fitted by least
squares from the 5 keypoints) warps every face into a canonical 112x112 crop. Not
affine or projective: those have enough freedom to deform the face, destroying the very geometry
that identifies a person. This is why single reference photos work as well as they do.

**Embedding (ResNet-50 + ArcFace, `w600k_r50.onnx`).** ArcFace's training loss normalises
features and class weights to unit length (so only the *angle* between them matters), then adds
an angular margin to the correct class before the softmax. The network isn't rewarded for being
right; it's rewarded for being right *with clearance*. Forcing 600k identities apart by a hard
margin makes it find features that separate human faces **in general**, which is why the space
still behaves for people it never trained on. That property is the whole foundation here.

**Matching.** Embeddings are L2-normalised, so cosine similarity is a plain dot product. A
person's score is the **max** over their reference photos, not the mean: references
span different angles and years, and averaging them produces a centroid resembling no actual
photograph. Max asks the right question: *does this match **any** photo I hold of you?*

**Clustering.** One person on screen for ten seconds yields ~25 detections. Asking a human "who
is this?" 25 times is not an interface, so detections are greedily grouped into apparent people
(cosine >= 0.5 to a cluster's representative). Each cluster keeps its clearest, largest detection
as the representative; that's the crop you're shown and the one that gets enrolled.

**The temporal rule.** Decide per cluster, not per frame: recognized iff **>=3 of some 5
consecutive observations** score >= 0.40. At ~0.4 s sampling that's a ~2 s window needing ~1.2 s
of presence. Per-frame decisions sat on a knife edge: one hard-angle frame came within 0.004 of
the threshold. The rule roughly doubles the safety margin at no cost in recall (see Results).

**Execution.** Both models are ONNX graphs on ONNX Runtime, preferring CoreML (Apple GPU/ANE),
falling back to CPU. The single biggest speed decision wasn't the provider: `buffalo_l` ships
five models and by default runs all of them on every face, including 106-point landmarks, 3D
landmarks and age/gender, none of which this project reads. Restricting it to detection +
recognition cut per-frame cost from **106 ms to 61 ms** on crowded footage with byte-identical
results.

Measured on an M1 Pro: `frame ~ 36 ms (detection, fixed) + 3.7 ms x faces (embedding)`, so 13-16
fps on crowded footage. The batch scan samples ~8 % of frames and runs **5.6-7.1x faster than
realtime**. Detection cost does not depend on video resolution, so the sampling interval, not
resolution, is the knob that matters.

---

## Run it

Python 3.12. ffmpeg is optional but recommended; without it YouTube downloads fall back to a
low-resolution progressive stream (smaller faces, worse recognition) and section downloads are
unavailable.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m streamlit run src/app.py     # run from the repo root
```

The face model (~182 MB) downloads itself on first use into `~/.insightface/models/buffalo_l/`.
On first run the gallery is empty, so start at **Add person**.

### The app

| Page | What it does |
| --- | --- |
| **Recognize** | paste a YouTube link (or pick/upload a file) -> downloads + scans -> names everyone it knows, asks about everyone it doesn't |
| **Live + learn** | paste a link -> **streams** with live recognition boxes, nothing downloaded -> Stop -> name the unknowns |
| **Add person** | type a name -> rights-clear candidate portraits from Wikimedia Commons, or your own uploads -> pick -> enrolled |
| **Gallery** | browse / rename / merge / delete enrolled people and individual reference photos |

**Recognize** is the main flow. Unknown faces come back as cards reading like
`3 frames · 1.2-4.5s · nearest Silva 0.38 · 3of5 0.35 < 0.40` reads as: seen in 3 sampled frames, closest
gallery person Ana Maria Silva at 0.38, sustained score below the threshold. Name one from the dropdown
(picking an *existing* person is how you add a missed frame rather than creating a duplicate) and
it's enrolled immediately. **Re-scan to verify** then reports exactly what changed:
*"Now recognized: İlkay Işık · unknown 4 -> 3."* That is the learning loop, closed, in about ten
seconds.

**Live + learn** shows one processed frame per tick: a slideshow, not smooth video. That's
inherent: recognition runs at a few frames per second, and anything smoother would mean not
recognizing most frames. For smooth playback use the standalone window below.

**Add person:** one or two good photos beat five mediocre ones. This isn't a style preference;
see the false-reject finding in Results.

### The standalone window

Watch-only: full detection on every frame, boxes and names **only** for recognized people.
Unknown faces get nothing: no box, no prompt, no learning. It opens a GUI window, so run it
yourself in a terminal.

```bash
.venv/bin/python src/live_recognition.py          # interactive: asks for a source each time
```

It asks what to watch (YouTube link / file / webcam), optionally records an annotated mp4, plays
to the end, then asks whether you want another. Model and gallery load once and are reused for
the whole session. `q`/`ESC` quits, `space` pauses.

```bash
.venv/bin/python src/live_recognition.py --video data/videos/video_01.mp4
.venv/bin/python src/live_recognition.py --video 0                        # webcam
.venv/bin/python src/live_recognition.py --video "https://youtu.be/XXXX"  # YouTube stream
```

This window decides **per frame**: it holds no cluster, so there's no series to aggregate and
the 3-of-5 rule doesn't apply. Labels react instantly and recover the moment a face turns back
toward the camera.

Rebuild the gallery from `data/reference_photos/<person>/*.jpg` (rare maintenance):

```bash
.venv/bin/python -c "import sys; sys.path.insert(0,'src'); import gallery as G; \
G.save(G.build_from_photos('data/reference_photos'), 'data/gallery.json')"
```

---

## Results

The one-off scripts that produced these numbers and their raw JSON output are not part of this
repository; they ship with the project report.

### Backend head-to-head

4 people / 26 reference photos, 88 genuine + 237 impostor pairs, all on CPU:

| Backend | genuine mean | impostor mean | worst-case gap | speed |
| --- | --- | --- | --- | --- |
| **InsightFace (buffalo_l)** | 0.759 | 0.053 | **+0.422** (clean) | **207 ms/img** |
| facenet-pytorch (vggface2) | 0.834 | 0.063 | +0.165 (clean) | 777 ms/img |
| DeepFace (ArcFace/RetinaFace) | 0.625 | 0.149 | **-0.070 (overlap)** | 2077 ms/img |

"Worst-case gap" is the hardest genuine pair minus the easiest impostor pair: what a single
global threshold actually has to survive, which mean separation flatters. DeepFace's
distributions genuinely overlap (easiest impostor 0.438 > hardest genuine 0.368), so no threshold
separates them cleanly here. InsightFace won on both axes at once: widest gap *and* 3.8x faster
than the next candidate.

### The 0.40 threshold

Photos did **not** choose it. On photos, genuine pairs run 0.587-0.957 and impostors stay <= 0.165,
a gap of +0.422 with nothing inside it, so any cutoff in ~0.20-0.55 is error-free. Video chose
it: across 130 observations on real 360p footage, scores spread much lower and wider (median
0.645, top 0.789), with the genuine cluster reaching down to ~0.40 as faces get small, blurred
and turned away. 0.40 sits at the bottom edge of that cluster. When it errs, it errs
toward a **false reject**.

### Held-out validation, and what a bigger gallery changed

Leave-one-out CV: hold out one embedding, rebuild the gallery without it, try to recognize it.
Run twice; the second run reflects the current gallery.

| | 4 people / 32 emb | **15 people / 64 emb** |
| --- | --- | --- |
| separation margin | +0.243 | **+0.048** |
| FAR @ 0.40 | 0 | **0** |
| FRR @ 0.40 | 0 | **13.3 %** |
| genuine min | 0.408 | 0.280 |
| impostor max | 0.165 | 0.233 |
| impostor pairs tested | 237 | **896** |

This says two different things, and the table alone misleads on both:

**False accepts: the evidence got stronger.** FAR is 0.0 at 0.40 and stays 0.0 across the entire
sweep from 0.24 upward, now over 896 impostor pairs including four public figures added
specifically to stress it. The highest any impostor pair reached is 0.233. The system saying
someone's name when it's somebody else remains unobserved.

**False rejects: a real regression with a locatable cause.** 8 of 60 genuine pairs now fall
below 0.40, and they are not spread evenly: 7 of the 8 belong to four people enrolled from small
video-frame crops with only 2-3 references each (they lose 3/3, 2/2, 1/5 and 1/3 of their
references); the eighth is one hard frame of the best-covered person (1 of 15). Everyone
enrolled from clean portraits still passes at 100 %. This is a
**reference-quality** problem, not a threshold problem; the fix is pruning tiny reference crops,
not moving the cutoff.

**Top-1 needs a caveat or it misreports.** The raw figure is 60/64, but all four "errors" are
people with exactly **one** reference photo: leave-one-out removes it, leaving nothing to match
against, so they're unmatchable by construction and scored as failures. On the 60 evaluable
pairs, top-1 is **60/60**.

**Why not just lower the threshold, if the sweep says 0.24-0.28 is error-free?** Because that
sweep is over photos, and photos never chose this threshold. On video, strangers reach 0.309 per
frame, so a 0.28 cutoff would admit them. The temporal rule is what actually resolves the tension.

### Video and the temporal rule

Four clips, one of them picked for having lots of strangers. Every borderline cluster scoring 0.30-0.40 was
exported as a bounding-box contact sheet (37 cards) and checked by eye. **Every single
near-miss was an already-enrolled person at a hard angle.** Not one was a stranger creeping
toward the threshold. Verified stranger ceiling: **0.309** per frame.

From the 16-rule sweep:

| | per-frame | 3-of-5 |
| --- | --- | --- |
| expected identities kept | 5 / 5 | 5 / 5 |
| false accepts | 0 | 0 |
| stranger ceiling | 0.309 | **0.191** |
| margin to threshold | 0.296 | **0.385** |

The margin roughly doubles at no cost in recall.

### Safeguards

Enrollment contamination is the failure that quietly destroys a gallery: label one face wrongly
and a stranger's embedding now sits under someone's name, so that stranger matches from then on,
silently, compounding. Four checks run before any enrollment (`review_unknowns.preflight`): the
crop must re-detect a face; a multi-face crop is flagged; the crop's dominant face must match the
clustered one (cosine >= 0.6); and the face must not be threshold-close to a *different* enrolled
person. Each is a warning with an explicit override: the human decides, but on purpose. Crops
are copied into `data/reference_photos/` before enrollment (so rebuilding reproduces the gallery)
and every decision is appended to `decisions.jsonl`.

---

## Limits

- **Small data.** 15 people, 64 embeddings, 4 clips. Every number means "no errors of this type
  observed on this data", not a bound. The impostor evidence is the strong part (896 pairs, zero
  false accepts); the genuine side is thin.
- **The gallery is not clean.** Some references are 27-36 px crops from video frames and carry
  most of the false-reject rate. No cross-person contamination was found (max cross-person cosine
  0.233), so the problem is weakness, not poisoning.
- **Threshold and margin are backend-bound.** 0.40 is a property of `buffalo_l` embeddings; a
  different backend invalidates it entirely.
- **Appearances under ~1.2 s cannot be auto-recognized** on the batch path, by construction.
- **Clustering is greedy, single-pass and order-dependent**: an early mistake is never revisited.
  The 0.5 cutoff is a heuristic, not calibrated.
- **No liveness or anti-spoofing.** A photograph held up to a camera would be embedded like a
  face. Out of scope for the research question; a real gap in any deployment reading.
- **CoreML pins the detector input to 640x640**: `det_10g.onnx` has 640-based shapes baked into
  its graph, so any other `det_size` crashes.

---

## Repository layout

```
src/            the runtime system
  app.py             multi-page Streamlit app (the entry point)
  gallery.py         core: InsightFace model + JSON gallery + matching
  collect_unknowns.py video scan: detect/match/cluster + the 3-of-5 decision
  review_unknowns.py labeling cards + enrollment safeguards (contamination guard, audit log)
  live_learn.py      the "Live + learn" streaming player page
  live_recognition.py real-time cv2 window
  video_library.py   YouTube download / tidy naming / stream-URL resolution (yt-dlp)
  wiki_faces.py      Wikimedia Commons portrait fetch for "Add person"
data/           reference photos, videos, gallery.json (git-ignored) (biometric data)
results/        scan outputs, experiment JSONs (git-ignored)
```

`gallery.py` is the dependency-free core and imports InsightFace **lazily**, so gallery I/O and
matching work without the model or its weights present.

The research scripts behind the Results section ship with the project report rather than with
this repository. One of them, `backend_comparison.py`, needs the two alternative backends
installed to run: `.venv/bin/pip install deepface facenet-pytorch` (kept out of the runtime env;
they add ~1.6 GB of TF/torch).

```bash
.venv/bin/pip install ruff pre-commit
.venv/bin/ruff check src
```

---

## Data, privacy & licensing

- **Biometric data stays local.** Reference photos, videos and the embedding gallery are all
  git-ignored and never leave the machine. Every enrollment decision is appended to an audit log
  (`decisions.jsonl`). KVKK (Turkish data-protection law) implications are discussed in the
  project report.
- **Image sources.** Reference portraits come from official / rights-clear sources (Wikimedia
  Commons via its official API, official presidential portraits), never watermarked stock
  (Getty/AFP). For photos you upload yourself, the licensing responsibility is yours.
- **Model license.** InsightFace's pretrained weights are for non-commercial research use, which
  covers this research project. Productizing would require revisiting it.
- Project code is MIT-licensed (see `LICENSE`); the InsightFace model weights are **not** covered
  by it.
