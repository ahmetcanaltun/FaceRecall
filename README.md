# Face Recognition in Video

Recognize known people in real, low-quality video
from **very few reference photos** (often a single one), and **learn new people on the fly**:
when the system sees a face it doesn't know, it asks who it is, enrolls the answer, and knows
them on the very next pass.

Built as a research project. The research findings (backend comparison,
threshold calibration, held-out validation, temporal decision rule) are the primary
deliverable; the Streamlit app is the working prototype that ties them together.

## How it works

```
video ──> SCRFD face detector ──> ArcFace embedder (512-d)          [InsightFace buffalo_l]
                                        │
                        cosine similarity vs. enrolled gallery      [data/gallery.json]
                                        │
                greedy per-person clustering across sampled frames
                                        │
              3-of-5 temporal rule at the calibrated 0.40 threshold
                                        │
            "recognized: silva (0.71)"  /  "unknown: who is this?"
```

- **No training, ever.** A pretrained ArcFace network maps every face to a 512-d embedding;
  adding a person = storing their embedding(s) in a JSON gallery. That's what makes the
  few-shot + learn-on-the-fly loop possible.
- **Decisions are per apparent person, not per frame:** a cluster (or live track) is
  recognized iff >=3 of some 5 consecutive sampled observations reach the 0.40 cosine
  threshold, a single hot/cold frame can't flip the outcome.
- **Live overlay** (the standalone window, `src/live_recognition.py`): every frame is detected
  and matched, and only recognized people get a box. A Kalman tracker between sparse detections
  was tried and dropped, on real news footage (camera motion, cuts) the predicted boxes drifted
  and the temporal rule's label lag read as a bug.

## Results (research evidence)

The one-off scripts that produced these numbers (`experiments/*.py`) and their raw JSON output
are not part of this repository, they are included with the project report. They are named
below so each figure can be traced to the run that produced it.

Backend head-to-head (4 people / 26 reference photos, 88 genuine + 237 impostor pairs, CPU,
`experiments/backend_comparison.py`):

| Backend | genuine mean | impostor mean | margin | EER FAR/FRR | speed |
| --- | --- | --- | --- | --- | --- |
| **InsightFace (buffalo_l)** | **0.759** | **0.053** | **+0.42 (clean)** | **0.0 / 0.0** | **207 ms/img** |
| facenet-pytorch (vggface2) | 0.834 | 0.063 | +0.17 (clean) | 0.0 / 0.0 | 777 ms/img |
| DeepFace (ArcFace/retinaface) | 0.625 | 0.149 | -0.07 (overlap) | 0.013 / 0.011 | 2077 ms/img |

Validation of the chosen backend + 0.40 threshold:

- **Held-out photos** (leave-one-out CV over 32 embeddings, `experiments/holdout_evaluation.py`):
  FAR = FRR = 0 at 0.40, top-1 accuracy 32/32, held-out margin +0.243.
- **Video** (4 low-quality YouTube clips incl. a stranger-heavy one,
  `experiments/video_false_accept_scan.py`): zero verified false accepts; every borderline
  score (0.30-0.40) was human-verified to be an already-enrolled person themselves at a hard angle (false
  rejects, not near-false-accepts). Verified stranger ceiling: 0.309 per frame.
- **Temporal 3-of-5 rule** (16-rule sweep, `experiments/temporal_aggregation.py`): keeps all
  expected identities, zero false accepts, and widens the video margin 0.296 -> 0.385.
- Runs on Apple GPU/ANE via the CoreML execution provider: 42 ms/img (4.7x CPU) with ~0.9998
  embedding parity.

Honest caveats: this is a small-data operating point (4 people, 4 clips), the claim is
"no errors observed", not a bound. The only failure mode seen is a false reject on very hard
frames. The clustering cutoff (0.5) is a heuristic, not calibrated.

## Run it

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m streamlit run src/app.py
```

The app opens in your browser with five pages:

| Page | What it does |
| --- | --- |
| **Recognize** | paste a YouTube link (or pick/upload a file) -> downloads + scans -> names everyone it knows, asks about everyone it doesn't |
| **Live + learn** | paste a link -> **streams** with live recognition boxes (no download) -> Stop -> name the unknowns |
| **Add person** | type a name -> rights-clear candidate portraits fetched from Wikimedia Commons -> pick -> enrolled |
| **Gallery** | browse / rename / merge / delete enrolled people |

There's also a standalone real-time window that only boxes + names the people it recognizes
and ignores everyone else, no learning. The source can be a file, a webcam (`--video 0`), or
a YouTube link (streamed live, nothing downloaded):

```bash
.venv/bin/python src/live_recognition.py          # interactive: asks for a source each time
```

It asks what to watch (link / file / webcam), plays it to the end, then asks whether you want
another one, the model and gallery load once and are reused for the whole session. Pass
`--video` to skip the questions and play a single source directly:

```bash
.venv/bin/python src/live_recognition.py --video data/videos/video_01.mp4
.venv/bin/python src/live_recognition.py --video 0                        # webcam
.venv/bin/python src/live_recognition.py --video "https://youtu.be/XXXX"  # YouTube stream
```

Rebuild the gallery from `data/reference_photos/<person>/*.jpg` (rare maintenance):

```bash
.venv/bin/python -c "import sys; sys.path.insert(0,'src'); import gallery as G; \
G.save(G.build_from_photos('data/reference_photos'), 'data/gallery.json')"
```

## Repository layout

```
src/            the runtime system
  app.py             multi-page Streamlit app (the entry point)
  gallery.py         core: InsightFace model + JSON gallery + matching
  collect_unknowns.py video scan: detect/match/cluster + the 3-of-5 decision
  review_unknowns.py labeling cards + enrollment safeguards (contamination guard, audit log)
  live_learn.py      the "Live + learn" streaming player page
  live_recognition.py real-time cv2 window demo
  video_library.py   YouTube download / tidy naming / stream-URL resolution (yt-dlp)
  wiki_faces.py      Wikimedia Commons portrait fetch for "Add person"
data/           reference photos, videos, gallery.json (git-ignored) (biometric data)
results/        scan outputs, experiment JSONs (git-ignored)
docs/           technology report (TR/EN) + pipeline figure
```

The research scripts behind the Results section ship with the project report rather than with
this repository. One of them, `backend_comparison.py`, also needs the two alternative backends
installed to run: `.venv/bin/pip install deepface facenet-pytorch` (kept out of the runtime
env, they add ~1.6 GB of TF/torch).

## Tooling

```bash
.venv/bin/pip install ruff pre-commit
.venv/bin/ruff check src
```

## Data, privacy & licensing

- **Biometric data stays local:** reference photos, videos, and the embedding gallery are all
  git-ignored. Every enrollment decision is appended to an audit log (`decisions.jsonl`).
  KVKK (Turkish data-protection law) implications are discussed in the project report.
- **Image sources:** reference portraits come from official/rights-clear sources (e.g.
  Wikimedia Commons via its API, official presidential portraits), never watermarked stock.
- **Model license:** InsightFace's pretrained models are for non-commercial research use,
  which covers this research project. Productizing would require revisiting this.
- Project code is MIT-licensed (see `LICENSE`); the InsightFace model weights are **not**
  covered by it.
