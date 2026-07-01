# Spot the Fake Photo

This repository implements a lightweight, high-accuracy binary classifier that distinguishes authentic real-world photos from screen or paper printout recaptures using hand-engineered classical computer vision features.

---

## Quickstart

### 1. Installation
Install the necessary dependencies in your virtual environment:
```bash
pip install -r requirements.txt
```

### 2. Run Inference
To test an image, run:
```bash
python predict.py path/to/image.jpg
```
This prints a single float value between `0.0` (authentic photo) and `1.0` (screen/printout recapture) to stdout, rounded to 4 decimal places.

### 3. Retrain Pipeline
To run the Leave-One-Session-Out (LOGO) cross-validation and retrain the final classifier:
```bash
python src/train.py
```
This outputs candidate classifier performance, sweeps PCA components, prints per-session accuracy metrics, and saves the final fitted Pipeline to `models/model.pkl` and configuration parameters to `models/metadata.json`.

### 4. Benchmark Latency
To run the computational latency benchmark:
```bash
python src/benchmark.py
```

---

## Image Preprocessing Details

To ensure consistency across diverse camera sensors, aspect ratios, and resolutions, we apply standardized preprocessing before feature extraction:
1. **RGB Standardization**: Images are loaded using PIL and explicitly converted to RGB mode to discard color profiles and alpha channels.
2. **Dimension Scaling**: For spatial filters (sharpness, color, geometry), the image is scaled so its maximum dimension is capped at 512 pixels, preserving the aspect ratio.
3. **Pixel-Grid Preservation**: For DCT block boundary detection, the image is converted to grayscale *without resizing* to keep the 8x8 compression boundaries pixel-aligned.
4. **Grayscale Standardization**: FFT-based spatial frequency checks (halftone print dot regularities, moiré patterns) are executed on resized 512x512 grayscale matrices to standardize the frequency bounds.

---

## Repository Structure

```text
spot-the-fake-photo/
├── predict.py          # Command-line entrypoint for single-image inference
├── requirements.txt    # Python library dependencies
├── NOTE.md             # Engineering write-up: Dataset, rationale, metrics, challenges, cost
├── README.md           # Project guide, quickstart, preprocessing rules
├── dataset/
│   ├── real/           # 47 authentic photos across 7 sessions
│   └── screen/         # 49 screen & printout recaptures across 7 sessions
├── src/
│   ├── features.py     # Classical 39-dimensional feature extraction engine
│   ├── train.py        # Model selection, LOGO CV validation, and exporter
│   └── benchmark.py    # Latency benchmarking utility
└── models/
    ├── model.pkl       # Trained Pipeline: StandardScaler -> RandomForestClassifier
    ├── metadata.json   # JSON file containing feature names, parameters, and accuracy scores
    └── feature_importance.png
```
