# Screen Recapture Detector

This repository implements a binary image classifier that distinguishes authentic real-world photos from screen or printout recaptures (spoofing attempts). It is designed to work with small, session-structured datasets by combining physically-grounded hand-engineered classical computer vision features with deep learning representations from a frozen pretrained network.

## Quickstart

Follow these steps to set up the environment and run a prediction on a sample image:

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate   # On Windows
source .venv/bin/activate # On Unix/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Predict recapture probability for a target image
python predict.py dataset/real/r1_household_day/r1_01.jpeg
```

On success, the script prints only a single probability float rounded to 4 decimal places (e.g. `0.1928`) to stdout, representing the probability that the image is a screen recapture or printout.

---

## Repo Structure

```text
├── dataset/
│   ├── real/             # Authentic photos grouped by session folders (r1_household_day, etc.)
│   ├── screen/           # Recaptures grouped by session folders (s1_laptop, etc.)
│   └── shot_log.csv      # Log detailing device, target display, lighting, and distance for each session
├── models/
│   ├── model.pkl         # Serialized final trained pipeline (scaler + PCA + Logistic Regression)
│   ├── metadata.json     # Configuration parameters, threshold, and performance metrics
│   └── feature_importance.png # Coefficients of the PCA components
├── src/
│   ├── benchmark.py      # Script to measure core prediction latency
│   ├── features.py       # Core feature extraction logic (classical + MobileNetV3 embeddings)
│   ├── predict.py        # CLI interface wrapping inference
│   └── train.py          # LOGO validation, threshold optimization, and retraining script
├── predict.py            # CLI wrapper at the project root
├── NOTE.md               # Technical report detailing metrics, latency, cost, and improvements
└── requirements.txt      # List of dependencies
```

---

## How It Works

The system utilizes a hybrid feature ensemble consisting of 14 classical computer vision features (capturing 2D FFT periodic moiré patterns, Lab gamut range, Laplacian sharpness variance, local binary pattern texture entropy, glare blobs, and Hough line geometry) concatenated with 576-dimensional frozen MobileNetV3-Small embeddings. This 590-dimensional hybrid representation is normalized and projected onto 50 PCA components to prevent overfitting. A regularized Logistic Regression classifier evaluates the components and outputs the final recapture probability. Full details on the mathematical and feature extraction layers are documented in [NOTE.md](NOTE.md).

---

## Reproducing Training

To re-run the training pipeline from scratch, execute:

```bash
python -m src.train
```

The script performs the following operations:
1. Scans `dataset/` and extracts the hybrid 590-dimensional feature vector for all images.
2. Compares four candidate models (Logistic Regression, Gradient Boosting, Linear SVM, and RBF SVM) under Leave-One-Session-Out (LOGO) cross-validation.
3. Conducts a grid search over PCA component sizes `[50, 75, 100]`.
4. Performs threshold optimization to maximize the F1-score on out-of-fold predictions.
5. Retrains the optimal pipeline configuration on the full dataset and writes updated outputs to `models/`.

---

## Validation Methodology

Leave-One-Session-Out (LOGO) cross-validation is used as the primary validation methodology. Rather than performing a random split, which suffers from severe session-level leakage (since pictures in the same session share identical devices, backgrounds, and lighting), each LOGO fold holds out an entire session folder for testing. This ensures the model is evaluated on its ability to generalize to unseen devices and capture environments—providing a robust approximation of real-world production performance.

---

## Repository Diagnostics

* **Supporting Documentation**: Refer to [NOTE.md](NOTE.md) for detailed evaluation metrics, latency benchmarks, and hosting cost calculations. Refer to [dataset/shot_log.csv](dataset/shot_log.csv) for data collection hardware details.
* **Dependencies**: Verification confirmed that all required packages (`numpy`, `opencv-python`, `scikit-learn`, `scikit-image`, `Pillow`, `joblib`, `matplotlib`, `torch`, `torchvision`) are correctly imported. The `pandas` library is listed in `requirements.txt` but is currently unused in the codebase.
* **TODOs / Placeholders**: No placeholder comments or TODO code blocks are present in the repository.
