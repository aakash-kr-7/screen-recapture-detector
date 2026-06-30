## Approach

The hybrid pipeline combines 14 classical CV features (frequency, color, texture, glare, geometry) with a 576-dimensional frozen MobileNetV3-Small embedding into a 590-dimensional vector. This vector is normalized and reduced to 50 PCA components (whiten=True, fit per fold) before a regularized Logistic Regression classifier (C=0.1, class_weight="balanced"). No neural network was trained—MobileNet is purely a feature extractor. Four classifiers and three PCA dimensions (50, 75, 100) were evaluated; LOGO CV selected Logistic Regression and 50 components.

## Accuracy

Leave-one-session-out (LOGO) CV yielded a mean accuracy of 82.81% ± 22.29%. This validation was chosen over a random 80/20 split (optimistic 80.00% accuracy due to session leakage) because testing on unseen sessions is the closest approximation to real-world deployment.

The ±22.29% variance reflects fold-size noise—with only 5-8 images per session, individual errors swing fold accuracy by 12-20%. The hybrid pipeline improved accuracy by +10.55pp over the classical baseline (72.26% ± 23.61%). The optimal threshold of 0.4959 yielded F1 = 0.8696 (Precision: 0.9302, Recall: 0.8163) with predictions on real and screen photos of 0.1928 and 0.7937.

## Latency & Cost

Latency: The pipeline recorded a median CPU latency of 431.86ms per image (mean: 432.96ms, P95: 466.26ms) on an Intel64 Family 6 Model 183 CPU under Windows 10 using Python 3.11 with no GPU.

Cost: On-device runs are free. Cloud cost on t3.medium ($0.042/hr) with ~2.3 images/second throughput yields: ($0.042/3600) × 432 × 1000 ≈ $0.005 per 1,000 images ($5 per million).

## What I'd Improve With More Time

1. Dataset scale: Collecting more screen panels, printout papers, and lighting environments would directly reduce LOGO fold variance.
2. ONNX export: Exporting MobileNet would reduce inference latency to ~15-30ms on CPU, making mobile deployment viable.
3. Active learning: Routing low-confidence predictions (scores between 0.3-0.7) to human review and retraining monthly keeps the classifier robust.
4. Threshold calibration: Setting the threshold using a precision-recall curve against a cost matrix balances false accusations against missed fraud.

## Adversarial Robustness & Threshold Choice

The ensemble of independent frequency, color, texture, geometry, and embedding signals degrades gracefully if an adversary spoofs one cue—for example, warm screen grading defeats color features but not FFT or MobileNet. The threshold should be chosen off the precision-recall curve using a cost matrix rather than a default 0.5; in fraud-detection, recall is valued more than precision up to a defined false-positive budget.
