# Spot the Fake Photo

This system extracts hand-engineered classical computer vision features and deep learning embeddings to perform binary classification distinguishing authentic real-world photos from screen or printout recaptures.

## How It Works

Screen recaptures leave distinct physical artifacts including frequency-domain periodic moiré patterns from display pixels, compressed color gamuts, specularity reflections on protective screen glass, and straight axis-aligned boundaries from device bezels or paper edges. Hand-engineered computer vision features capture these specific physical anomalies by extracting 2D Fast Fourier Transform magnitude distributions, CIELAB gamut standard deviations, Laplacian edge-sharpness statistics, Local Binary Pattern micro-texture entropies, specular glare blob boundaries, and horizontal/vertical Hough line alignments.

Frozen MobileNetV3-Small ImageNet embeddings are concatenated with the classical features to add dense semantic representations of color, glare, and spatial structures that classical CV logic might not explicitly capture. The combined 590-dimensional hybrid representation is normalized and projected onto 50 PCA dimensions to avoid overfitting before passing to a regularized Logistic Regression classifier that predicts the recapture probability.

## Results

| Metric | Value |
|---|---|
| Validation method | Leave-one-session-out CV (14 sessions) |
| LOGO accuracy | 82.81% ± 22.29% |
| Optimal threshold F1 | 0.8696 (Precision: 0.9302, Recall: 0.8163) |
| Inference latency (CPU) | 432ms median (Intel Raptor Lake, no GPU) |
| Cost per 1,000 images (cloud) | ~$0.005 (t3.medium assumption) |
| Cost per 1,000 images (on-device) | Free |

See NOTE.md for full methodology, cost arithmetic, and improvement roadmap.

## Quickstart

### Install
```bash
pip install -r requirements.txt
```

### Run inference
```bash
python predict.py path/to/image.jpg
```
Prints a single float: 0 = real photo, 1 = photo of a screen.

### Retrain from scratch
```bash
python src/train.py
```
Requires dataset/real/ and dataset/screen/ to be populated. Saves model artifacts to models/.

### Benchmark latency
```bash
python src/benchmark.py
```

## Repository Structure

```text
spot-the-fake-photo/
├── predict.py          # entry point — run this
├── requirements.txt
├── NOTE.md             # submission write-up: approach, accuracy, cost
├── README.md
├── dataset/
│   ├── real/           # 47 real photos across 7 sessions
│   └── screen/         # 49 screen/printout photos across 7 sessions
├── src/
│   ├── features.py     # 590-dim feature extraction (14 classical + 576 MobileNet)
│   ├── train.py        # LOGO CV, model selection, artifact export
│   ├── predict.py      # inference function (called by root predict.py)
│   └── benchmark.py    # latency measurement
└── models/
    ├── model.pkl       # trained pipeline: StandardScaler → PCA(50) → LogisticRegression
    ├── metadata.json   # training metadata and evaluation metrics
    └── feature_importance.png
```

## Validation Methodology

Leave-one-session-out (LOGO) cross-validation is used as the evaluation methodology instead of a random K-fold split to prevent session-level data leakage. A "session" represents a unique capture environment with a specific device, lighting profile, and scene layout. Because images in the same session share identical backgrounds and hardware sensors, a random split would cause leakage between training and testing sets, yielding misleadingly high accuracy metrics. The ±22.29% standard deviation reflects fold-size noise, as holding out a single session with only 5-8 images means each classification error alters the fold accuracy by 12-20 percentage points rather than indicating true model instability.
