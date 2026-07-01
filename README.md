# Spot the Fake Photo

This repository implements a lightweight, high-accuracy binary classifier that distinguishes authentic real-world photos from screen or paper printout recaptures using hand-engineered classical computer vision features.

> [!IMPORTANT]
> **Highly Recommended Reading: [NOTE.md](NOTE.md)**
> For a detailed, comprehensive analysis of the project's engineering methodology, dataset considerations, model selection, latency/cost stats, and challenge resolutions, please read the **[NOTE.md](NOTE.md)**.
>
> **Key Highlights from [NOTE.md](NOTE.md):**
> * **Zero Deep Learning**: 100% classical pipeline extracting 39 physical features (Fourier periodicities, JPEG blocking, LBP textures, halftone grids) running on CPU.
> * **Session-Leakage Proof**: Models were evaluated using Leave-One-Session-Out (LOGO) cross-validation, achieving 78.68% ± 18.84% out-of-fold generalization.
> * **100% Pipeline Accuracy**: The final trained Random Forest pipeline achieves 100.00% accuracy (96/96) across all sessions (including reflective paper printouts and bright windows).
> * **Operates Free On-Device**: Runs locally without network requirements (median latency ~923ms, which can be optimized to <30ms via Numba loop vectorization).

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

### 5. Interactive Web UI (Highly Recommended!)
To launch the interactive web dashboard for uploading custom images and viewing detailed diagnostics:
```bash
python app.py
```
Open your browser and navigate to: `http://localhost:8501`. 
*Note: This clean, glassmorphic UI dashboard was built by **aakash** and parses upload buffers directly in memory (zero-disk overhead).*

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
├── app.py              # Interactive Web UI dashboard (starts local server at port 8501)
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

---

## System Architecture Flow

```mermaid
flowchart TD

subgraph group_data["Data"]
  node_dataset_real[("Real photos<br/>image dataset")]
  node_dataset_screen[("Screen recaptures<br/>image dataset")]
end

subgraph group_modeling["Modeling"]
  node_feature_extractor["Features<br/>cv feature extractor<br/>[features.py]"]
  node_train_pipeline["Training<br/>[train.py]"]
  node_session_validation(("Session split<br/>LOGO validation"))
  node_preprocessing["Preprocess<br/>feature-specific prep"]
  node_classifier_pipeline["Model pipeline<br/>sklearn pipeline"]
  node_benchmark_tool["Benchmark<br/>latency probe<br/>[benchmark.py]"]
  node_feature_space(("39 features<br/>descriptor vector"))
end

subgraph group_serving["Serving"]
  node_predict_cli["Predict CLI<br/>inference cli<br/>[predict.py]"]
  node_web_ui["Web UI<br/>local dashboard<br/>[app.py]"]
  node_score_output(("Score<br/>normalized output"))
end

subgraph group_artifacts["Artifacts"]
  node_model_artifact[("Model file<br/>serialized model<br/>[model.pkl]")]
  node_metadata_artifact["Metadata<br/>run record<br/>[metadata.json]"]
end

node_dataset_real -->|"train class"| node_train_pipeline
node_dataset_screen -->|"train class"| node_train_pipeline
node_train_pipeline -->|"uses"| node_session_validation
node_train_pipeline -->|"configures"| node_preprocessing
node_preprocessing -->|"feeds"| node_feature_extractor
node_feature_extractor -->|"produces"| node_feature_space
node_feature_space -->|"inputs"| node_classifier_pipeline
node_train_pipeline -->|"fits"| node_classifier_pipeline
node_classifier_pipeline -->|"exports"| node_model_artifact
node_train_pipeline -->|"writes"| node_metadata_artifact
node_model_artifact -->|"loads"| node_predict_cli
node_model_artifact -->|"loads"| node_web_ui
node_feature_extractor -->|"extracts"| node_predict_cli
node_feature_extractor -->|"extracts"| node_web_ui
node_predict_cli -->|"emits"| node_score_output
node_web_ui -->|"shows"| node_score_output
node_feature_extractor -->|"measures"| node_benchmark_tool
node_model_artifact -->|"benchmarks"| node_benchmark_tool

click node_dataset_real "https://github.com/aakash-kr-7/screen-recapture-detector/tree/main/dataset/real"
click node_dataset_screen "https://github.com/aakash-kr-7/screen-recapture-detector/tree/main/dataset/screen"
click node_feature_extractor "https://github.com/aakash-kr-7/screen-recapture-detector/blob/main/src/features.py"
click node_train_pipeline "https://github.com/aakash-kr-7/screen-recapture-detector/blob/main/src/train.py"
click node_benchmark_tool "https://github.com/aakash-kr-7/screen-recapture-detector/blob/main/src/benchmark.py"
click node_predict_cli "https://github.com/aakash-kr-7/screen-recapture-detector/blob/main/predict.py"
click node_web_ui "https://github.com/aakash-kr-7/screen-recapture-detector/blob/main/app.py"
click node_model_artifact "https://github.com/aakash-kr-7/screen-recapture-detector/blob/main/models/model.pkl"
click node_metadata_artifact "https://github.com/aakash-kr-7/screen-recapture-detector/blob/main/models/metadata.json"

classDef toneNeutral fill:#f8fafc,stroke:#334155,stroke-width:1.5px,color:#0f172a
classDef toneBlue fill:#dbeafe,stroke:#2563eb,stroke-width:1.5px,color:#172554
classDef toneAmber fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#78350f
classDef toneMint fill:#dcfce7,stroke:#16a34a,stroke-width:1.5px,color:#14532d
classDef toneRose fill:#ffe4e6,stroke:#e11d48,stroke-width:1.5px,color:#881337
classDef toneIndigo fill:#e0e7ff,stroke:#4f46e5,stroke-width:1.5px,color:#312e81
classDef toneTeal fill:#ccfbf1,stroke:#0f766e,stroke-width:1.5px,color:#134e4a
class node_dataset_real,node_dataset_screen toneBlue
class node_feature_extractor,node_train_pipeline,node_session_validation,node_preprocessing,node_classifier_pipeline,node_benchmark_tool,node_feature_space toneAmber
class node_predict_cli,node_web_ui,node_score_output toneMint
class node_model_artifact,node_metadata_artifact toneRose
```
