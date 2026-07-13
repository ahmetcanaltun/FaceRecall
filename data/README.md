# Data folder

This folder holds real reference photos/videos of actual people. It is
**not committed to git** (see `.gitignore`), biometric data of identifiable
individuals shouldn't live in version control. The enrolled gallery
(`gallery.json`, embedding-derived biometric data) is git-ignored for the same
reason.

## Layout

```
data/
  gallery.json             # the enrolled gallery (people -> 512-d embeddings + metadata)
  reference_photos/
    <person_name>/         # one folder per person, e.g. ana_maria_silva/
      photo1.jpg
      photo2.jpg
  videos/
    video_NN.mp4           # tidy sequential names; real titles live in library.json
    library.json           # index: file -> source URL / title / recognized people
    uploads/               # videos uploaded through the app
```

## Naming convention

- Person folder names: lowercase, ASCII, words separated by underscores
  (e.g. `ali_yildirim`, not `Ali Yıldırım`), the same snake_case keys the
  gallery uses. The app normalizes typed names automatically.
- Drop in whatever photos you have per person, even a single photo is fine
  (the whole system is built for the few-shot scenario). More is better where
  available. Prefer official / rights-clear sources; avoid watermarked stock.
- Video files: any common container (mp4/mkv/mov/webm) OpenCV can read.

Everything here is read/written by the app (`streamlit run src/app.py`); see
the root README for the gallery-rebuild one-liner if you need to regenerate
`gallery.json` from `reference_photos/` by hand.
