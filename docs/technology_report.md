# Yüz Tanıma Teknolojisi ve Tekniği, Bilgilendirme Raporu
# Face Recognition Technology & Technique, Informative Report

> Bu rapor, `data/gallery.json` içindeki her yüz için üretilen **512 sayılık gömme
> vektörlerini (embedding)** hangi teknolojinin ve hangi tekniğin ürettiğini açıklar.
> This report explains which **technology** and which **technique** produce the
> **512-number embedding vectors** stored for each face in `data/gallery.json`.
>
> Not / Note: Backend seçimi (InsightFace/buffalo_l) 2026-07-13'te, Task 6 doğrulamasının
> ardından kullanıcı onayıyla **resmen finalize edildi**. / The backend choice
> (InsightFace/buffalo_l) was **formally finalized** on 2026-07-13, with user sign-off,
> after the Task 6 validation.

---

## 🇹🇷 TÜRKÇE

### 1. Kısa Özet

Her yüz, bir yapay sinir ağı tarafından **512 adet sayıdan oluşan bir vektöre** dönüştürülür.
Bu vektöre **gömme (embedding)** denir ve yüzün "kimlik parmak izi" gibidir. Aynı kişinin
farklı fotoğrafları birbirine **yakın** vektörler; farklı kişiler birbirinden **uzak**
vektörler üretir. Tanıma işlemi, yeni bir yüzün vektörünün, veritabanındaki bilinen bir
yüzün vektörüne ne kadar yakın olduğunu (kosinüs benzerliği) ölçmekten ibarettir.

Bu vektörleri üreten teknoloji **InsightFace** kütüphanesi ve onun **buffalo_l** model
paketidir. Kullanılan temel teknik ise **ArcFace** (derin metrik öğrenme) yöntemidir.

### 2. Kullanılan Teknoloji

**Köken** sütunu, bileşenin hazır mı geldiğini yoksa bu projede mi yazıldığını gösterir,
sistemin neresi kütüphane çağrısı, neresi proje katkısı sorusunun cevabıdır.

| Katman | Bileşen | Köken | Bu projedeki dosya / model |
| --- | --- | --- | --- |
| Kütüphane | **InsightFace** (açık kaynak yüz analiz kütüphanesi) | hazır | `insightface` (Python) |
| Model paketi | **buffalo_l** | önceden eğitilmiş | `~/.insightface/models/buffalo_l/` |
| Yüz tespiti (detection) | **SCRFD-10GF** | önceden eğitilmiş | `det_10g.onnx` (16 MB) |
| Yüz tanıma / gömme (recognition) | **ResNet-50 + ArcFace**, WebFace600K ile eğitilmiş | önceden eğitilmiş | `w600k_r50.onnx` (166 MB) |
| Çalışma motoru (inference) | **ONNX Runtime** (bu projede CoreML, Apple GPU/ANE; macOS dışında CPU'ya düşer; CPU'ya göre ~4.7x hızlı, gömme eşdeğerliği ~0.9998 kosinüs) | hazır | `CoreMLExecutionProvider` -> `CPUExecutionProvider` |
| Sayısal altyapı | **NumPy**, tüm vektör matematiği (L2 normalizasyon, iç çarpım, matris çarpımı) | hazır | `numpy` |
| Görüntü G/Ç | **OpenCV**, kare okuma, kırpma, çizim | hazır | `cv2` |
| Arayüz | **Streamlit**, çok sayfalı web uygulaması | hazır | `streamlit` |
| Video kaynağı | **yt-dlp**, YouTube bağlantısını doğrudan akış URL'sine çözer (indirme yok) | hazır | `yt-dlp` |
| İzleme (araştırma izi) | **norfair**, Kalman filtresiyle seyrek tespitler arası kutu taşıma | hazır | `norfair` (§8; **çalışan sistemde kullanılmıyor**) |
| **Eşleştirme ve karar katmanı** | kosinüs benzerliği, kişi-başına en-yakın-referans, 0.40 eşiği, açgözlü çevrimiçi kümeleme, 3-of-5 zamansal kural, kayıt (enrollment) korumaları ve denetim kaydı | **bu projede yazıldı** | `src/gallery.py`, `src/collect_unknowns.py`, `src/review_unknowns.py` |
| **Uygulama ve akışlar** | çok sayfalı arayüz, canlı pencere, video kütüphanesi, Wikimedia/yükleme ile kişi ekleme | **bu projede yazıldı** | `src/app.py`, `src/live_recognition.py`, `src/video_library.py`, `src/wiki_faces.py` |
| **Deneyler / kanıt üretimi** | arka uç karşılaştırması, eşik kalibrasyonu, held-out doğrulama, video yanlış-kabul taraması, zamansal kural süpürmesi | **bu projede yazıldı** | `experiments/*.py` -> `results/*.json` |

> **Önemli doğruluk notu:** `buffalo_l` paketinde yüz **tespiti** için kullanılan model
> **SCRFD**'dir (`det_10g.onnx`), RetinaFace değil. RetinaFace, InsightFace'in bir başka
> (kardeş) dedektörüdür ama bu pakette yer almaz.
>
> Pakette ayrıca `2d106det.onnx` (106 noktalı yüz işaret tespiti), `1k3d68.onnx` (3B
> işaret noktaları) ve `genderage.onnx` (yaş/cinsiyet) modelleri de vardır; **bu projede
> tanıma için bunları kullanmıyoruz**, sadece dedektör ve tanıma modelini kullanıyoruz.

### 3. Kullanılan Teknik / Yöntem

**a) Derin Metrik Öğrenme (Deep Metric Learning), gömme (embedding) yaklaşımı**
Klasik yöntemde her kişi için ayrı bir "sınıf" tanımlanır ve ağ "bu an enrolled person mı?" diye
sınıflandırma yapacak şekilde eğitilir. Yeni kişi eklendiğinde ağın yeniden eğitilmesi
gerekir. Bunun yerine metrik öğrenmede ağ, yüzleri **anlamlı bir vektör uzayına** yerleştirmeyi
öğrenir. Böylece yeni kişi eklemek = sadece o kişinin vektörünü veritabanına kaydetmek;
**yeniden eğitim gerekmez.** Az veriyle (hatta tek fotoğrafla) çalışabilmesinin nedeni budur.

**b) ArcFace (Additive Angular Margin Loss)**
`w600k_r50` modeli, **ArcFace** kayıp fonksiyonu ile eğitilmiştir (Deng vd., CVPR 2019).
ArcFace, eğitim sırasında farklı kimlikleri bir **hiperküre (birim küre) üzerinde açısal
olarak** birbirinden olabildiğince ayırır. Sonuç: aynı kişinin vektörleri dar bir açıyla
kümelenirken, farklı kişiler geniş açılarla ayrılır. Vektörler **L2 ile normalize edildiği**
(uzunlukları 1 yapıldığı) için, iki yüzü karşılaştırmanın doğal yolu **kosinüs benzerliğidir**
(iki vektör arasındaki açının kosinüsü; 0 = alakasız, 1 = birebir aynı).

**c) 512 boyutlu gömme (embedding)**
Ağın çıkışı, her yüz için **512 sayıdan** oluşan sabit uzunlukta bir vektördür. Bu sayılar
tek tek "burun genişliği", "göz rengi" gibi insanca okunabilir şeyler **değildir**; ağın
milyonlarca yüz üzerinde öğrendiği soyut özelliklerdir. Anlam, sayıların **hep birlikte**
oluşturduğu konumdadır ve asıl önemli olan **vektörler arasındaki mesafedir.**

**d) Karşılaştırma metriği: Kosinüs benzerliği (Öklid DEĞİL)**
Yüz gömmeleri için Öklid mesafesi yerine **kosinüs benzerliği** kullanılır. Örnek (bu
projenin kendi verisinden):
- an enrolled person foto-0 ↔ an enrolled person foto-1: **0.851** (aynı kişi -> yakın)
- an enrolled person ↔ Lars Petersen: **0.123** (farklı kişi -> uzak)
- an enrolled person ↔ Nina Kovac: **0.044** (farklı kişi -> uzak)

### 4. İşlem Hattı (Bir Fotoğraf -> 512 Sayı)

```
1. Fotoğrafı oku            (OpenCV / cv2.imread)
2. Yüzü tespit et           (SCRFD / det_10g.onnx) → kutu + 5 işaret noktası
3. Yüzü hizala (align)      → 5 işaret noktasıyla 112×112 standart görüntüye dönüştür
4. Gömme çıkar              (ResNet-50 + ArcFace / w600k_r50.onnx) → 512 sayı
5. L2 normalize et          → vektör uzunluğu 1 (kosinüs karşılaştırması için)
6. gallery.json'a kaydet    → {src, emb[512], det_score, face_px, date, flags}
```

> Not: 3. adımdaki "hizalama" sırasında InsightFace bir benzerlik dönüşümü (similarity
> transform) hesaplar; çalıştırırken gördüğümüz `estimate`/`lstsq` uyarıları tam olarak bu
> adımdan gelir ve zararsızdır.

### 5. Neden Bu Yaklaşım? (Projenin Bağlamı)

- **Az veri:** Bazı kişiler için tek referans fotoğraf var. Metrik öğrenme + gömme
  yaklaşımı tek örnekle bile çalışır (few-shot / açık küme tanıma).
- **Yeniden eğitim yok:** Yeni bir kişi eklemek, sadece bir JSON kaydı eklemektir.
- **Kanıt:** Düşük kaliteli (360p) gerçek videolarda bilinen kişiler doğru tanındı,
  yabancılarda yanlış kabul (false accept) gözlenmedi.

### 6. `gallery.json` ile İlişkisi

Yukarıdaki 6 adımlık işlem hattı, her referans fotoğraf için **bir kez** çalıştırılır ve
sonuç `data/gallery.json` içine yazılır (kişi başına birden çok gömme + meta veri). Sonraki
çalışmalar fotoğrafları yeniden işlemez; doğrudan bu dosyayı okur. Dosya **biyometrik türevli
veri** içerdiği için `.gitignore` ile sürüm kontrolünün dışında tutulur (KVKK).

### 7. Arka Uç Karşılaştırması (Task 2)

Üç aday kütüphane, aynı 26 referans fotoğraf üzerinde (88 aynı-kişi + 237 farklı-kişi çifti,
kosinüs benzerliği, CPU) karşılaştırıldı:

| Arka uç | aynı-kişi ort. | farklı-kişi ort. | ayrışma | hız |
| --- | --- | --- | --- | --- |
| **InsightFace (buffalo_l)** | 0.759 | 0.053 | **+0.42 (temiz)** | **207 ms/foto** |
| facenet-pytorch (vggface2) | 0.834 | 0.063 | +0.17 (temiz) | 777 ms/foto |
| DeepFace (ArcFace/retinaface) | 0.625 | 0.149 | -0.07 (ÖRTÜŞME) | 2077 ms/foto |

InsightFace hem en geniş ayrışmayı verdi hem de en hızlısıydı; projenin geri kalanı bu arka
uçla ilerledi ve seçim, Task 6 doğrulamasının ardından **2026-07-13'te kullanıcı onayıyla
finalize edildi** (bu tablo + §8-9 o kararın kanıt tabanıdır).

### 8. Eşik Kalibrasyonu ve Bağımsız Doğrulama (Task 4 + 6)

- **Eşik: 0.40 kosinüs** (kullanıcı onayı 2026-07-03). Fotoğraf çiftleri temiz ayrışır
  (aynı-kişi 0.587-0.957, farklı-kişi <= 0.164; EER ~ %0), bağlayıcı kısıt videodur.
- **Bağımsız (held-out) doğrulama** (`holdout_evaluation.py`, 32 gömme üzerinde birini-dışarıda-
  bırak çapraz doğrulama): 0.40'ta **FAR = FRR = 0**, en-yakın-kişi doğruluğu **32/32**, kişi
  başına %100; galeri kirliliği yok (şüpheli `nina_kovac_4` fotoğrafı kontrol edildi, temiz).
  Dürüstlük notu: örneklem-içi "+0.42" ayrışma, held-out ölçümde **+0.243**'e iner ve aynı-kişi
  tabanı (0.408) eşiğin hemen üstündedir, pay fotoğrafta bile sanıldığı kadar geniş değildir.

### 9. Videoda Davranış ve Zamansal Karar Kuralı (Task 6, 2026-07-13)

Kare-başına karar videoda **ince bir payla** çalışır: 4 klipte eşik-altı kümeler 0.396-0.397'ye
kadar çıktı. Sınırdaki her kümenin insan gözüyle doğrulanması (37 kartlık, kutu-çizili kontrol
sayfası; `results/temporal_aggregation_verification.json`) belirleyici bulguyu verdi:

> **0.30-0.40 aralığındaki "neredeyse-kabul" vakalarının tamamı, aslında galerideki kişinin
> kendisinin zor kareleriydi** (profil, kapanma/occlusion, hareket bulanıklığı), yani bunlar
> yanlış-kabul riski değil, **yanlış-red** vakalarıdır. Gerçek yabancıların doğrulanmış tavanı
> kare-başına **0.309**'dur ve 4 klipte **sıfır yanlış kabul** gözlendi.

İnce payın çözümü eşiği oynatmak değil, **kararı kare yerine küme (görünen kişi) düzeyinde
vermek**: `temporal_aggregation.py`, 16 kuralı taradı (k-of-n, pencere ortalaması vb.). Kabul edilen
kural (kullanıcı kararı 2026-07-13): **5 ardışık gözlemden >=3'ü >= 0.40 ise tanı** ("3-of-5";
~0.4 sn örnekleme ile ~2 sn'lik pencere).

| | kare-başına | **3-of-5 (kabul edilen)** |
| --- | --- | --- |
| Tanınan kimlikler (4 klip) | 5/5 | 5/5 |
| Yanlış kabul | 0 | 0 |
| Yabancı tavanı | 0.309 | **0.191** |
| Etkin pay | 0.296 | **0.385** |

Bilinen bedel: ~1.2 sn'den kısa görünen kişi doğrulanamaz (toplu etiketleme uygulaması için
kabul edilebilir; canlı yayın kaplaması yapılırsa yeniden değerlendirilmeli). Bu deney ayrıca
"takip (tracking) şart mı?" sorusunun **doğruluk** yarısını da yanıtladı: gerçek bir IoU
takipçisine gerek kalmadan, mevcut kümeler sözde-iz (pseudo-track) olarak yeterli.

### 10. Sınırlılıklar (Dürüstlük Bölümü)

- **Küçük veri:** 4 kişi, 26-32 gömme, 4 klip. Bütün sayılar "bu veride hata gözlenmedi"
  ifadesidir; kanıtlanmış bir sınır değildir.
- **Galeri büyüdükçe:** yabancı tavanının 100+ kişilik galeride nasıl davranacağı ölçülmedi
  (ölçek deneyi kapsam dışı bırakıldı). En yakın komşu benzerliği galeri büyüdükçe yükselme
  eğilimindedir; büyük galeri hedefleniyorsa yeniden kalibrasyon gerekir.
- **Küme parçalanması:** açgözlü kümeleme (0.5 kosinüs) aynı kişiyi bir klipte birden çok
  kümeye bölebiliyor (ör. bir klipte Petersen 19 parçada). Kimlik düzeyinde sonuç değişmedi ama
  1-2 gözlemlik parçalar 3-of-5 kuralı altında tek başına doğrulanamaz.
- **KVKK:** `gallery.json` biyometrik türevli veri içerir (git dışı tutulur); her etiketleme
  kararı `decisions.jsonl` denetim izine yazılır. Hukuki değerlendirme proje raporunun ayrı
  bölümüdür.

---

## 🇬🇧 ENGLISH

### 1. Executive Summary

Each face is converted by a deep neural network into a **vector of 512 numbers**, called an
**embedding**, effectively the face's "identity fingerprint." Different photos of the *same*
person produce **nearby** vectors; *different* people produce **far-apart** vectors.
Recognition is simply measuring how close a new face's vector is to a known face's vector in
the database (via cosine similarity).

The technology that produces these vectors is the **InsightFace** library with its
**buffalo_l** model pack. The core technique is **ArcFace** (deep metric learning).

### 2. Technology Used

The **Origin** column says whether a component came off the shelf or was written for this
project, i.e. which part of the system is a library call and which part is the project's own
contribution.

| Layer | Component | Origin | File / model in this project |
| --- | --- | --- | --- |
| Library | **InsightFace** (open-source face analysis toolkit) | off the shelf | `insightface` (Python) |
| Model pack | **buffalo_l** | pretrained | `~/.insightface/models/buffalo_l/` |
| Face detection | **SCRFD-10GF** | pretrained | `det_10g.onnx` (16 MB) |
| Face recognition / embedding | **ResNet-50 + ArcFace**, trained on WebFace600K | pretrained | `w600k_r50.onnx` (166 MB) |
| Inference engine | **ONNX Runtime** (CoreML, Apple GPU/ANE in this project; falls back to CPU off-macOS; ~4.7x faster than CPU with ~0.9998 cosine embedding parity) | off the shelf | `CoreMLExecutionProvider` -> `CPUExecutionProvider` |
| Numerics | **NumPy**, all vector math (L2 normalisation, dot product, matrix multiply) | off the shelf | `numpy` |
| Image I/O | **OpenCV**, frame reading, cropping, drawing | off the shelf | `cv2` |
| UI | **Streamlit**, multi-page web app | off the shelf | `streamlit` |
| Video source | **yt-dlp**, resolves a YouTube link to a direct stream URL (no download) | off the shelf | `yt-dlp` |
| Tracking (research trail) | **norfair**, Kalman box carrying between sparse detections | off the shelf | `norfair` (§8; **not used in the running system**) |
| **Matching & decision layer** | cosine similarity, per-person nearest reference, the 0.40 threshold, greedy online clustering, the 3-of-5 temporal rule, enrollment guards and the audit log | **written for this project** | `src/gallery.py`, `src/collect_unknowns.py`, `src/review_unknowns.py` |
| **Application & flows** | multi-page UI, live window, video library, add-person via Wikimedia/uploads | **written for this project** | `src/app.py`, `src/live_recognition.py`, `src/video_library.py`, `src/wiki_faces.py` |
| **Experiments / evidence** | backend comparison, threshold calibration, held-out validation, video false-accept scan, temporal rule sweep | **written for this project** | `experiments/*.py` -> `results/*.json` |

> **Accuracy note:** In the `buffalo_l` pack the face **detector** is **SCRFD**
> (`det_10g.onnx`), *not* RetinaFace. RetinaFace is a sibling InsightFace detector but is not
> part of this pack.
>
> The pack also contains `2d106det.onnx` (106-point landmarks), `1k3d68.onnx` (3D
> landmarks) and `genderage.onnx` (age/gender), but **we do not use these for recognition**,
> only the detector and the recognition model are used.

### 3. Technique / Method

**a) Deep Metric Learning, the embedding approach**
A classic classifier defines one "class" per person and is trained to answer "is this
an enrolled person?"; adding a new person requires retraining. Metric learning instead trains the
network to place faces into a **meaningful vector space**. Adding a new person then means
simply storing that person's vector, **no retraining.** This is exactly why it works with
little data (even a single photo).

**b) ArcFace (Additive Angular Margin Loss)**
The `w600k_r50` model was trained with the **ArcFace** loss (Deng et al., CVPR 2019). During
training, ArcFace pushes different identities as far apart as possible **angularly on a unit
hypersphere.** The result: the same person's vectors cluster within a narrow angle, while
different people are separated by wide angles. Because the vectors are **L2-normalized** (unit
length), the natural way to compare two faces is **cosine similarity** (the cosine of the
angle between the two vectors; 0 = unrelated, 1 = identical).

**c) The 512-dimensional embedding**
The network's output is a fixed-length vector of **512 numbers** per face. These numbers are
**not** individually human-readable features (there is no single "nose width" number); they
are abstract features learned over millions of faces. The meaning lives in the vector **as a
whole**, and what matters is the **distance between vectors.**

**d) Comparison metric: cosine similarity (not Euclidean)**
For face embeddings we use **cosine similarity**, not Euclidean distance. Example (from this
project's own data):
- an enrolled person photo-0 ↔ an enrolled person photo-1: **0.851** (same person -> close)
- an enrolled person ↔ Lars Petersen: **0.123** (different -> far)
- an enrolled person ↔ Nina Kovac: **0.044** (different -> far)

### 4. The Pipeline (One Photo -> 512 Numbers)

```
1. Read the image           (OpenCV / cv2.imread)
2. Detect the face          (SCRFD / det_10g.onnx) → bounding box + 5 landmarks
3. Align the face           → warp to a canonical 112×112 crop using the 5 landmarks
4. Extract the embedding    (ResNet-50 + ArcFace / w600k_r50.onnx) → 512 numbers
5. L2-normalize             → vector length 1 (so cosine comparison is well-defined)
6. Store in gallery.json    → {src, emb[512], det_score, face_px, date, flags}
```

> Note: the "alignment" in step 3 computes a similarity transform; the `estimate`/`lstsq`
> warnings seen at runtime come precisely from this step and are harmless.

### 5. Why This Approach? (Project Context)

- **Scarce data:** some people have only one reference photo. Metric learning + embeddings
  work even with a single example (few-shot / open-set recognition).
- **No retraining:** enrolling a new person is just adding one JSON record.
- **Evidence:** on genuinely low-quality (360p) real videos, known people were recognized
  correctly and no false accepts were observed on strangers.

### 6. Relationship to `gallery.json`

The 6-step pipeline above is run **once** per reference photo, and the result is written into
`data/gallery.json` (multiple embeddings per person plus metadata). Later runs do not
re-process the photos; they read this file directly. Because the file contains
**biometric-derived data**, it is kept out of version control via `.gitignore` (KVKK).

### 7. Backend Comparison (Task 2)

Three candidate libraries were compared on the same 26 reference photos (88 genuine + 237
impostor pairs, cosine similarity, CPU):

| Backend | genuine mean | impostor mean | separation | speed |
| --- | --- | --- | --- | --- |
| **InsightFace (buffalo_l)** | 0.759 | 0.053 | **+0.42 (clean)** | **207 ms/img** |
| facenet-pytorch (vggface2) | 0.834 | 0.063 | +0.17 (clean) | 777 ms/img |
| DeepFace (ArcFace/retinaface) | 0.625 | 0.149 | -0.07 (OVERLAP) | 2077 ms/img |

InsightFace gave both the widest separation and the highest speed; the rest of the project
proceeded on it, and the choice was **formally finalized on 2026-07-13 with user sign-off**
after the Task 6 validation (this table plus §8-9 form that decision's evidence base).

### 8. Threshold Calibration and Held-Out Validation (Tasks 4 + 6)

- **Threshold: 0.40 cosine** (user sign-off 2026-07-03). Photo pairs separate cleanly
  (genuine 0.587-0.957, impostor <= 0.164; EER ~ 0%), video is the binding constraint.
- **Held-out validation** (`holdout_evaluation.py`, leave-one-out cross-validation over the 32
  enrolled embeddings): **FAR = FRR = 0** at 0.40, top-1 identification **32/32**, 100% per
  person; no enrollment contamination (the flagged `nina_kovac_4` photo was checked and is
  benign). Honesty note: the in-sample "+0.42" separation shrinks to **+0.243** held-out, and
  the genuine floor (0.408) sits just above the threshold, the margin is narrower than the
  in-sample numbers suggested, even on photos.

### 9. Behaviour on Video and the Temporal Decision Rule (Task 6, 2026-07-13)

Per-frame decisions run on a **thin margin** on video: across 4 clips, sub-threshold clusters
reached 0.396-0.397. Human verification of every borderline cluster (a 37-card contact sheet
with the scoring face boxed in the full frame; `results/temporal_aggregation_verification.json`)
produced the decisive finding:

> **Every "almost-accepted" case in the 0.30-0.40 band is actually the gallery person
> themselves in a hard frame** (profile, occlusion, motion blur), i.e. these are **false
> rejects**, not near false-accepts. The verified per-frame ceiling for actual strangers is
> **0.309**, and **zero false accepts** were observed across all 4 clips.

The fix for the thin margin is not moving the threshold but **deciding per cluster (apparent
person) instead of per frame**: `temporal_aggregation.py` swept 16 rules (k-of-n, window mean, …).
The adopted rule (user decision 2026-07-13): recognize iff **>=3 of any 5 consecutive
observations score >= 0.40** ("3-of-5"; a ~2 s window at the ~0.4 s sampling interval).

| | per-frame | **3-of-5 (adopted)** |
| --- | --- | --- |
| Identities recognized (4 clips) | 5/5 | 5/5 |
| False accepts | 0 | 0 |
| Stranger ceiling | 0.309 | **0.191** |
| Effective margin | 0.296 | **0.385** |

Known cost: a person on screen for less than ~1.2 s cannot be confirmed (acceptable for the
batch labeling app; revisit for any live overlay). This experiment also answered the
**accuracy** half of the "is tracking a must?" question: no real IoU tracker was needed, the
existing clusters serve as pseudo-tracks.

### 10. Limitations (Honesty Section)

- **Small data:** 4 people, 26-32 embeddings, 4 clips. Every number here means "no errors
  observed on this data", not a proven bound.
- **Gallery growth:** how the stranger ceiling behaves with a 100+ person gallery was not
  measured (the scale experiment was descoped). Nearest-neighbour similarity tends to rise as
  the gallery grows; re-calibrate before scaling up.
- **Cluster fragmentation:** the greedy clustering (0.5 cosine) can split one person into many
  clusters within a clip (e.g. Petersen into 19 fragments in one clip). Identity-level results
  were unaffected, but 1-2 observation fragments can never be confirmed alone under 3-of-5.
- **KVKK:** `gallery.json` holds biometric-derived data (kept out of version control); every
  labeling decision is appended to the `decisions.jsonl` audit trail. The legal assessment is
  a separate section of the project report.

---

## Kaynaklar / References

- **ArcFace:** J. Deng, J. Guo, N. Xue, S. Zafeiriou, *"ArcFace: Additive Angular Margin Loss
  for Deep Face Recognition,"* CVPR 2019.
- **SCRFD:** J. Guo et al., *"Sample and Computation Redistribution for Efficient Face
  Detection,"* 2021 (InsightFace `det_10g` detector).
- **InsightFace:** https://github.com/deepinsight/insightface, açık kaynak / open source.
- **ONNX Runtime:** https://onnxruntime.ai, model çalıştırma motoru / inference engine.
- **buffalo_l model pack:** `w600k_r50` (ResNet-50, WebFace600K, ArcFace) + `det_10g` (SCRFD).

> Lisans notu / License note: InsightFace önceden eğitilmiş modeller **akademik/araştırma**
> kullanımı içindir. **Çözüm (2026-07-13, kapsam netleşmesiyle):** bu çalışma kurumun
> kullanacağı bir ürün değil, bir staj araştırma projesidir; dolayısıyla araştırma lisansı bu
> kullanımı kapsar. Üretime taşınacak olsaydı yeniden değerlendirilmesi gerekirdi (planlanmıyor).
> / The InsightFace pretrained models are for **academic/research** use. **Resolved
> (2026-07-13, by scope):** this work is a research project, not a product the organisation
> will run, so the research license covers this use. It would need revisiting only if the work
> were ever productized (not planned).
