# Take-Home Note: Spot the Fake Photo

This document details the engineering methodology, dataset considerations, model selection rationale, metrics, and challenges faced while building the classical computer vision screen recapture detector.

---

## 1. Dataset Considerations & Preprocessing

### The Dataset
I compiled a balanced dataset of **96 images** split into 14 distinct capture sessions (7 real, 7 screen) to simulate diverse environments:
* **Authentic Photos (47 instances)**:
  * `r1_household_day` (9): Indoor environments under natural daylight.
  * `r2_household_artificial` (8): Indoor environments under warm bulb/LED artificial lighting.
  * `r3_outdoor_day` (7): Outdoor daylight scenes.
  * `r4_confounders_bright` (7): Real scenes containing highly saturated bright light sources (windows, lamps).
  * `r5_confounders_glossy` (5): Real glossy surfaces exhibiting specular reflections.
  * `r6_people` (6): Photos containing human faces and skin tones.
  * `r7_textures` (5): Close-up texture details (wood grain, fabrics).
* **Recaptured Images (49 instances)**:
  * `s1_laptop` (10), `s3_monitor_bright` (7), `s4_tv_livingroom` (6), `s5_tablet` (5), `s6_phone_of_phone` (7): Screen recaptures across multiple emissive display panels.
  * `s2_miscellaneous_closeups` (7): Close-up angles highlighting pixel structures.
  * `s7_printout` (7): Physical printed paper recaptures (reflective ink/toner media, not emissive displays).

### Preprocessing Engineering
Standardizing inputs across different camera aspect ratios and resolutions was critical:
* **CIELAB Color Space**: Images are converted to Lab space. The luminance channel ($L$) is isolated to measure surface reflectance and highlight clipping independently of global color temperature.
* **HSV Saturation Gating**: Saturation is evaluated in HSV space. The saturation clipping features are gated: pixels are counted only if saturation $> 220$ and luminance $L \in [40, 230]$. This filters out pure white highlights (e.g. bright sky) which trigger false screen-saturation classifications.
* **High-Frequency Extraction**: Images are grayscaled and high-pass filtered via Gaussian blur subtraction to create a high-frequency noise residual.
* **Sobel Gradients & Hough Transforms**: Horizontal and vertical Sobel filters calculate pixel-level gradient orientations. Probabilistic Hough Transform isolates axis-aligned straight lines to find screen bezels and paper boundaries.

---

## 2. Engineering Approach & Way of Thinking

My core philosophy was to **avoid large black-box deep learning models** (like MobileNet embeddings). Neural networks introduce massive package footprints, GPU dependency, slow startup times, and opaque failure modes.

Instead, I focused on a **hybrid classical feature extraction** approach, modeling the physical differences in how light behaves in real scenes vs. recaptures. I hand-engineered **39 features** spanning 8 distinct visual signatures:
1. **subpixel Periodic Moire**: RGB channels are evaluated in 2D FFT shifted space to find correlated high-frequency peaks caused by camera sensor pixels overlapping display subpixel stripes.
2. **JPEG quantization edges**: JPEG double-compression block edges create periodic gradient discontinuities at 8-pixel boundaries.
3. **Focus Homogeneity**: Camera lenses shooting a flat emissive display panel produce a highly uniform sharpness distribution. Real 3D scenes have depth-of-field variations and focus roll-offs.
4. **Gated Saturation & Uniformity**: Backlit displays exhibit clipped saturation peaks and high spatial saturation variance.
5. **Noise Kurtosis**: Camera sensors capturing active screens produce higher noise energy and distinct noise frequency distributions.
6. **LBP Micro-textures**: Local Binary Patterns capture pixel grid micro-grids.
7. **Straight Line Boundaries**: Displays and printed sheets have sharp, straight, axis-aligned borders.
8. **Reflective Paper Halftones**: Printed paper exhibits high mean reflectance ($L$), lack of deep blacks, and regular halftone dot patterns.

---

## 3. Model Selection & Rationale

I evaluated 5 candidate classifiers under **Leave-One-Session-Out (LOGO) cross-validation** with **5x training-only photometric augmentation** (brightness shifts and color balance variations generated on the fly inside the fold loop):

* **Why LOGO CV?** A standard random 80/20 train/test split suffers from session leakage. Images in the same session share background structures and sensor noise. Random splits yield a misleadingly high (~98%) validation accuracy but fail in production. LOGO CV holds out an entire session during testing, providing the only honest estimate of generalization to unseen environments.
* **Why Random Forest?** Random Forest outperforms linear models (Logistic Regression) and SVMs on our classical feature set. Classical features are highly heterogeneous (ratios, entropies, variances, counts). Decision trees naturally handle non-linear boundaries, heterogeneous dimensions, and are robust to outlier ranges without requiring delicate hyperparameter tuning.
* **LOGO CV Results**:
  * *Logistic Regression ($C=1.0$)*: 75.27% mean accuracy
  * *Random Forest*: **78.68% ± 18.84%** mean accuracy (Selected)
  * *Extra Trees*: 77.26% mean accuracy
  * *SVM (RBF Kernel)*: 67.00% mean accuracy
* **Final Trained Pipeline**: The final model (`models/model.pkl`) is fitted on the entire dataset (96 images × 5 variants = 480 augmented instances) using a standard scaler and RandomForest classifier. It achieves **100.00% training classification accuracy** (96/96 images correctly classified).

---

## 4. Challenges Faced & Solutions

### Challenge 1: Emissive-biased feature blindness to printed papers
Printed recaptures (`s7_printout`) are reflective, not emissive. They do not clip backlight saturation and lack moiré grids.
* **Solution**: I implemented **Group 8 Printout Features**. Halftone printing prints color dots at regularly spaced angles (e.g. 45°/75°). By applying a 2D FFT on a resized 512x512 grayscale matrix and **excluding horizontal/vertical axes by 20 degrees**, I isolated the circular variance of the remaining peaks (`print_halftone_angular_regularity`). Printouts produce regular peaks (variance $\approx 0.004$), while real photos and emissive screens produce scattered peaks (variance $> 0.15$).

### Challenge 2: Bright windows/sky causing false positives
Real photos containing bright light sources (`r4_confounders_bright`) clipped saturation values globally, tricking the baseline screen detector.
* **Solution**: Gated the saturation clipping check to only count pixels where HSV Saturation $> 220$ and CIELAB Luminance $L \in [40, 230]$. This successfully ignores bright sky and specular glare. I also added `color_sat_spatial_uniformity` (coefficient of variation of patch saturation), as screen displays exhibit non-uniform vivid regions, whereas natural skies clip uniformly.

### Challenge 3: Periodic natural textures misclassified as screen grids
Close-up texture photos (`r7_textures` like woven fabrics and fine wood grain) triggered DCT blocking and moiré periodicity metrics.
* **Solution**: Real periodic textures have gradient directions that continue smoothly across block boundaries. Screen recaptures with double compression exhibit gradient direction discontinuities *specifically* at the 8-pixel boundaries. I implemented `dct_gradient_direction_change`, which computes the angular difference between dominant Sobel gradients in the 2 rows above vs. 2 rows below the 8-pixel boundaries.

---

## 5. Latency & Cost

* **Inference Latency**:
  * Tested on an **Intel64 Family 6 Model 183 CPU** under Windows 10 using Python 3.11 with no GPU.
  * **Mean Latency**: **928.17 ms** per image.
  * **Median Latency**: **923.90 ms** per image.
  * **P95 Latency**: **952.09 ms** per image.
* **Operational Cost**:
  * **On-device**: **Free**. The pipeline runs entirely on CPU with zero network dependencies.
  * **Cloud instance**: Running on a t3.medium ($0.042/hr) with ~1.08 images/sec throughput:
    $$\text{Cost per 1,000 images} = \frac{\$0.042}{3600} \times 0.928 \times 1000 \approx \$0.011 \quad (\approx \$10.82 \text{ per million images})$$

---

## 6. What I'd Improve With More Time

* **Loop Vectorization (Numba / Cython)**: 
  Currently, the dominant computational latency (~800ms) is caused by Python-level loops in `_extract_dct_block_artifacts` that iterate over row/column indices to measure boundary gradient alignment. Compiling these loops with **Numba** or rewriting them in vectorized numpy block slicing operations would reduce latency from **~920ms to <30ms** on CPU.
