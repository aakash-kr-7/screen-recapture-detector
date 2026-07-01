"""
src/train.py - Model Training and Validation Pipeline

What it is:
  This script executes our model selection, validation, and training workflow. It loads the dataset, 
  extracts 39-dimensional classical features, performs Leave-One-Session-Out (LOGO) cross-validation, 
  and exports the final fitted Pipeline ('models/model.pkl') along with its 'metadata.json'.

Why I did what I did:
  1. Leave-One-Session-Out (LOGO) CV: A standard random train/test split suffers from massive data leakage 
     because photos taken in the same environment (same lighting, sensor, background) share signatures. 
     Evaluating on held-out sessions is the only way to get a realistic estimate of real-world accuracy.
  2. Training-Only Augmentation: We augment images inside the training folds by generating 4 photometric 
     variants (brightness shifts and color temperature adjustments) on the fly. This prevents validation 
     leakage and expands the size of our training set to make the classifier robust to diverse lighting.
  3. Classifier & PCA sweeps: We compare multiple models (Logistic Regression, Random Forest, SVM, etc.) 
     and PCA components to select the pipeline that generalizes best without overfitting.
"""

import os
import sys
import json
import datetime
import joblib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

# Import feature extractor and standard names
from src.features import extract_features, FEATURE_NAMES

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve


# ── DATA LOADING & AUGMENTATION ──────────────────────────────────────────────

def extract_all_variants(img_path: str) -> list[np.ndarray]:
    """
    Extracts features from the original image and its 4 photometric variants:
      0: Original features
      1: Brightness +40 (PIL enhance 1.3)
      2: Brightness -40 (PIL enhance 0.7)
      3: Warm white balance (R * 1.15, B * 0.85)
      4: Cool white balance (R * 0.85, B * 1.15)
    """
    with Image.open(img_path) as pil_img:
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
            
        # Original features
        feats_orig = extract_features(pil_img)
        
        # a) Brightness +40: PIL ImageEnhance.Brightness(img).enhance(1.3)
        enhancer = ImageEnhance.Brightness(pil_img)
        img_bright_plus = enhancer.enhance(1.3) # 1.3: factor for bright augmentation (+30% luminance)
        feats_bright_plus = extract_features(img_bright_plus)
        
        # b) Brightness -40: PIL ImageEnhance.Brightness(img).enhance(0.7)
        img_bright_minus = enhancer.enhance(0.7) # 0.7: factor for dim augmentation (-30% luminance)
        feats_bright_minus = extract_features(img_bright_minus)
        
        # Convert to numpy for WB scaling
        arr = np.array(pil_img, dtype=np.float32)
        
        # c) Warm white balance: R channel by 1.15, B channel by 0.85, clip to 255
        warm_arr = arr.copy()
        warm_arr[:, :, 0] *= 1.15 # 1.15: boost red channel for warmer tone
        warm_arr[:, :, 2] *= 0.85 # 0.85: scale down blue channel
        warm_arr = np.clip(warm_arr, 0, 255).astype(np.uint8) # 255: max value for 8-bit scale
        feats_warm = extract_features(warm_arr)
        
        # d) Cool white balance: R channel by 0.85, B channel by 1.15, clip to 255
        cool_arr = arr.copy()
        cool_arr[:, :, 0] *= 0.85 # 0.85: scale down red channel
        cool_arr[:, :, 2] *= 1.15 # 1.15: boost blue channel for cooler tone
        cool_arr = np.clip(cool_arr, 0, 255).astype(np.uint8) # 255: max value for 8-bit scale
        feats_cool = extract_features(cool_arr)
        
        return [feats_orig, feats_bright_plus, feats_bright_minus, feats_warm, feats_cool]


def load_dataset() -> tuple[np.ndarray, list[list[np.ndarray]], np.ndarray, np.ndarray, list[str]]:
    """
    Scans dataset/real/ and dataset/screen/ for images grouped by sessions.
    Extracts features for original and 4 augmented variants for each image.
    """
    real_dir = os.path.join("dataset", "real")
    screen_dir = os.path.join("dataset", "screen")
    
    sessions = []
    
    def scan_class_dir(class_dir: str, label: int):
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist.")
            return
        for session_name in sorted(os.listdir(class_dir)):
            session_path = os.path.join(class_dir, session_name)
            if os.path.isdir(session_path):
                image_files = []
                for entry in sorted(os.listdir(session_path)):
                    if entry.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_files.append(os.path.join(session_path, entry))
                if image_files:
                    sessions.append({
                        "name": session_name,
                        "label": label,
                        "images": image_files
                    })
                    
    scan_class_dir(real_dir, 0)
    scan_class_dir(screen_dir, 1)
    
    # Summary of sessions loaded
    print("\n" + "=" * 80)
    print(f"{'Session Name':<35} | {'Class':<8} | {'Image Count':<12}")
    print("=" * 80)
    for sess in sessions:
        label_str = "real" if sess["label"] == 0 else "screen"
        print(f"{sess['name']:<35} | {label_str:<8} | {len(sess['images']):<12}")
    print("=" * 80 + "\n")
    
    X_orig = []
    X_aug_list = []
    y = []
    groups = []
    paths = []
    
    total_images = sum(len(s["images"]) for s in sessions)
    processed = 0
    
    print(f"Extracting 32 classical features (original + 4 augmented variants) from {total_images} images...")
    for sess in sessions:
        for img_path in sess["images"]:
            try:
                variants = extract_all_variants(img_path)
                X_orig.append(variants[0])
                X_aug_list.append(variants[1:])
                y.append(sess["label"])
                groups.append(sess["name"])
                paths.append(img_path)
            except Exception as e:
                print(f"Warning: Skipping unreadable file '{img_path}' due to error: {e}")
            processed += 1
            if processed % 15 == 0 or processed == total_images:
                print(f"  Processed {processed}/{total_images} images...")
                
    return np.array(X_orig), X_aug_list, np.array(y), np.array(groups), paths


# ── LOGO CV LOOP ──────────────────────────────────────────────────────────────

def evaluate_pipeline(X_orig: np.ndarray, X_aug_list: list[list[np.ndarray]], y: np.ndarray, groups: np.ndarray, 
                      clf_class, clf_kwargs: dict, n_components) -> tuple[list[dict], np.ndarray]:
    """
    Evaluates a specific Pipeline configuration using Leave-One-Session-Out CV.
    Augments training folds ONLY; test folds remain original and unmodified.
    """
    logo = LeaveOneGroupOut()
    folds = []
    oof_probs = np.zeros(len(y))
    
    for train_idx, test_idx in logo.split(X_orig, y, groups):
        # Build training set (with 5x augmentation: original + 4 variants)
        X_train_list = []
        y_train_list = []
        for idx in train_idx:
            X_train_list.append(X_orig[idx])
            X_train_list.extend(X_aug_list[idx])
            y_train_list.extend([y[idx]] * 5) # 5: original + 4 variants
        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)
        
        # Test fold uses original unmodified features ONLY
        X_test = X_orig[test_idx]
        y_test = y[test_idx]
        
        # Print diagnostic note for the first fold split to confirm logic
        if len(folds) == 0:
            n_orig = len(train_idx)
            print(f"  Training fold augmented: {n_orig} original -> {n_orig}x5 = {n_orig*5} samples") # 5: original + 4 variants
            
        # Build Pipeline steps
        steps = [('scaler', StandardScaler())]
        if n_components is not None:
            n_samples = X_train.shape[0]
            n_features = X_train.shape[1]
            actual_npc = min(n_components, n_samples - 1, n_features)
            steps.append(('pca', PCA(n_components=actual_npc, whiten=True, random_state=42)))
        steps.append(('clf', clf_class(**clf_kwargs)))
        
        pipe = Pipeline(steps)
        pipe.fit(X_train, y_train)
        
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        oof_probs[test_idx] = y_prob
        
        folds.append({
            "session": groups[test_idx[0]],
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0)
        })
        
    return folds, oof_probs


# ── THRESHOLD SELECTION ───────────────────────────────────────────────────────

def select_threshold(y: np.ndarray, oof_probs: np.ndarray) -> float:
    """
    Computes a precision-recall curve and selects the decision threshold maximizing F1.
    """
    precisions, recalls, thresholds = precision_recall_curve(y, oof_probs)
    
    # Calculate F1 scores. Note thresholds shape is (N-1,) so align size.
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
    
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])
    best_f1 = f1_scores[best_idx]
    
    # Get values at default 0.5 threshold
    idx_05 = np.argmin(np.abs(thresholds - 0.5)) # 0.5: default threshold
    prec_05 = precisions[idx_05]
    rec_05 = recalls[idx_05]
    f1_05 = 2 * prec_05 * rec_05 / (prec_05 + rec_05 + 1e-8)
    
    print("\n" + "=" * 50)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 50)
    print(f"At default threshold 0.5: Precision={prec_05:.4f}, Recall={rec_05:.4f}, F1={f1_05:.4f}")
    print(f"Optimal threshold maximizing F1: {best_threshold:.4f}")
    print(f"  Max F1-Score:   {best_f1:.4f}")
    print(f"  Precision:      {precisions[best_idx]:.4f}")
    print(f"  Recall:         {recalls[best_idx]:.4f}")
    print("=" * 50 + "\n")
    
    return best_threshold


# ── FINAL RETRAINING & ARTIFACT SAVING ────────────────────────────────────────

def save_artifacts(X_orig: np.ndarray, X_aug_list: list[list[np.ndarray]], y: np.ndarray, groups: np.ndarray,
                   chosen_model_name: str, chosen_clf_class, chosen_clf_kwargs: dict, 
                   best_threshold: float, folds: list[dict], 
                   optimal_n_components, oof_probs: np.ndarray):
    """
    Retrains the chosen model class on the entire augmented dataset and exports files to models/.
    """
    # Build full augmented training dataset
    X_full_list = []
    y_full_list = []
    for idx in range(len(y)):
        X_full_list.append(X_orig[idx])
        X_full_list.extend(X_aug_list[idx])
        y_full_list.extend([y[idx]] * 5) # 5: original + 4 variants
    X_full_aug = np.array(X_full_list)
    y_full_aug = np.array(y_full_list)
    
    # Build final pipeline
    steps = [('scaler', StandardScaler())]
    if optimal_n_components is not None:
        n_samples_full = X_full_aug.shape[0]
        n_features_full = X_full_aug.shape[1]
        actual_npc_full = min(optimal_n_components, n_samples_full - 1, n_features_full)
        steps.append(('pca', PCA(n_components=actual_npc_full, whiten=True, random_state=42)))
    steps.append(('clf', chosen_clf_class(**chosen_clf_kwargs)))
    
    final_pipeline = Pipeline(steps)
    final_pipeline.fit(X_full_aug, y_full_aug)
    
    os.makedirs("models", exist_ok=True)
    
    # Save trained pipeline (includes scaler, PCA mapping if used, and classifier)
    model_path = os.path.join("models", "model.pkl")
    joblib.dump(final_pipeline, model_path)
    print(f"Saved final trained model pipeline to '{model_path}'")
    
    # Compute LOGO accuracies
    logo_accs = [f['accuracy'] for f in folds]
    mean_logo_acc = float(np.mean(logo_accs))
    std_logo_acc = float(np.std(logo_accs))
    
    # Calculate out-of-fold statistics at optimal threshold
    oof_preds = (oof_probs >= best_threshold).astype(int)
    best_prec = float(precision_score(y, oof_preds, zero_division=0))
    best_rec = float(recall_score(y, oof_preds, zero_division=0))
    best_f1 = float(f1_score(y, oof_preds, zero_division=0))
    
    # Save metadata.json
    metadata = {
        "model_type": chosen_model_name,
        "feature_count": 39,
        "augmentation": "4x photometric variants (brightness ±, warm/cool WB) on training only",
        "mobilenet_used": False,
        "classical_only": True,
        "pca_components": optimal_n_components,
        "logo_mean_accuracy": mean_logo_acc,
        "logo_std_accuracy": std_logo_acc,
        "optimal_threshold": best_threshold,
        "optimal_threshold_precision": best_prec,
        "optimal_threshold_recall": best_rec,
        "optimal_threshold_f1": best_f1,
        "total_image_count": int(len(y)),
        "real_image_count": int(np.sum(y == 0)),
        "screen_image_count": int(np.sum(y == 1)),
        "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature_names": FEATURE_NAMES
    }
    
    metadata_path = os.path.join("models", "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved model configuration metadata to '{metadata_path}'")
    
    # Plot feature importances
    plt.figure(figsize=(12, 10))
    clf_obj = final_pipeline.named_steps['clf']
    
    if "Forest" in chosen_model_name or "Trees" in chosen_model_name:
        importances = clf_obj.feature_importances_
        sorted_indices = np.argsort(importances)
        y_pos = np.arange(len(importances))
        plt.barh(y_pos, importances[sorted_indices], align='center', alpha=0.8, color='skyblue')
        plt.yticks(y_pos, [FEATURE_NAMES[idx] for idx in sorted_indices], fontsize=8)
        plt.xlabel("Importance Score")
        plt.title(f"{chosen_model_name} Feature Importances")
        
    elif "Logistic" in chosen_model_name:
        # Plot absolute coefficient values per feature
        if 'pca' in final_pipeline.named_steps:
            pca_obj = final_pipeline.named_steps['pca']
            # Map coefficients in PCA space back to original 32-dimensional feature space
            coef_feature_space = clf_obj.coef_[0] @ pca_obj.components_
            importances = np.abs(coef_feature_space)
            title = "Logistic Regression Feature Importance (Absolute Coefficients Mapped Back from PCA)"
        else:
            importances = np.abs(clf_obj.coef_[0])
            title = "Logistic Regression Feature Importance (Absolute Coefficients)"
            
        sorted_indices = np.argsort(importances)
        y_pos = np.arange(len(importances))
        plt.barh(y_pos, importances[sorted_indices], align='center', alpha=0.8, color='skyblue')
        plt.yticks(y_pos, [FEATURE_NAMES[idx] for idx in sorted_indices], fontsize=8)
        plt.xlabel("Absolute Coefficient Magnitude")
        plt.title(title)
        
    else: # SVM RBF
        # Plot explained variance ratio of PCA components if PCA was used
        if 'pca' in final_pipeline.named_steps:
            pca_obj = final_pipeline.named_steps['pca']
            importances = pca_obj.explained_variance_ratio_
            sorted_indices = np.argsort(importances)
            y_pos = np.arange(len(importances))
            plt.barh(y_pos, importances[sorted_indices], align='center', alpha=0.8, color='lightgreen')
            pca_names = [f"pca_component_{i:02d}" for i in range(len(importances))]
            plt.yticks(y_pos, [pca_names[idx] for idx in sorted_indices], fontsize=8)
            plt.xlabel("Explained Variance Ratio")
            plt.title("PCA Explained Variance (SVM Pipeline)")
        else:
            plt.text(0.5, 0.5, "SVM with raw features: no direct importance available", 
                     ha='center', va='center', fontsize=12, style='italic')
            plt.title("SVM Feature Importance")
            plt.axis('off')
            
    plt.tight_layout()
    plot_path = os.path.join("models", "feature_importance.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved feature importance visualization to '{plot_path}'")


# ── MAIN WORKFLOW ─────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("BINARY IMAGE CLASSIFIER CLASSICAL TUNING WORKFLOW")
    print("=" * 80)
    
    # 1. Load dataset with original and augmented features
    try:
        X_orig, X_aug_list, y, groups, _ = load_dataset()
    except Exception as exc:
        print(f"Error during data loading/feature extraction: {exc}", file=sys.stderr)
        sys.exit(1)
        
    if len(y) == 0:
        print("Error: No images found or feature extraction failed for all images.", file=sys.stderr)
        sys.exit(1)
        
    print(f"\nDataset loaded. Total instances: {len(y)}, Real count: {np.sum(y == 0)}, Screen count: {np.sum(y == 1)}")
    
    # 2. Run validations, compare classifiers under LOGO CV (with PCA = None as baseline)
    candidates = [
        {
            "name": "Logistic Regression (C=0.1)",
            "class": LogisticRegression,
            "kwargs": {"C": 0.1, "class_weight": "balanced", "max_iter": 1000, "random_state": 42}
        },
        {
            "name": "Logistic Regression (C=1.0)",
            "class": LogisticRegression,
            "kwargs": {"C": 1.0, "class_weight": "balanced", "max_iter": 1000, "random_state": 42}
        },
        {
            "name": "Random Forest",
            "class": RandomForestClassifier,
            "kwargs": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2, "class_weight": "balanced", "random_state": 42}
        },
        {
            "name": "Extra Trees",
            "class": ExtraTreesClassifier,
            "kwargs": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2, "class_weight": "balanced", "random_state": 42}
        },
        {
            "name": "SVM (RBF Kernel, C=10)",
            "class": SVC,
            "kwargs": {"kernel": "rbf", "C": 10, "gamma": "scale", "class_weight": "balanced", "probability": True, "random_state": 42}
        }
    ]
    
    print("\nCommencing Leave-One-Session-Out (LOGO) Candidate Comparisons (PCA=None/Raw)...")
    candidate_results = []
    for cand in candidates:
        folds, oof_probs = evaluate_pipeline(X_orig, X_aug_list, y, groups, cand["class"], cand["kwargs"], n_components=None)
        accs = [f['accuracy'] for f in folds]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        
        candidate_results.append({
            "name": cand["name"],
            "class": cand["class"],
            "kwargs": cand["kwargs"],
            "folds": folds,
            "oof_probs": oof_probs,
            "mean_acc": mean_acc,
            "std_acc": std_acc
        })
        
    # Print classifier comparison table
    print("\n" + "=" * 80)
    print("CLASSIFIER COMPARISON (PCA components = None/Raw)")
    print("=" * 80)
    print(f"{'Classifier Model':<35} | {'Mean LOGO Acc':<15} | {'Std Dev':<10}")
    print("-" * 80)
    for res in candidate_results:
        print(f"{res['name']:<35} | {res['mean_acc']*100:.2f}%         | {res['std_acc']*100:.2f}%")
    print("=" * 80 + "\n")
    
    # Choose best classifier based on mean LOGO accuracy
    best_cand = max(candidate_results, key=lambda x: x["mean_acc"])
    print(f"Selected Best Classifier: {best_cand['name']} (Mean Acc: {best_cand['mean_acc']*100:.2f}%)\n")
    
    # 3. PCA n_components sweep for the best classifier
    pca_options = [None, 20, 32, 50]
    print(f"Running PCA components sweep for {best_cand['name']} across [None, 20, 32, 50]...")
    pca_results = []
    for npc in pca_options:
        folds, oof_probs = evaluate_pipeline(X_orig, X_aug_list, y, groups, best_cand["class"], best_cand["kwargs"], n_components=npc)
        accs = [f['accuracy'] for f in folds]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        pca_results.append({
            "n_components": npc,
            "folds": folds,
            "oof_probs": oof_probs,
            "mean_acc": mean_acc,
            "std_acc": std_acc
        })
        
    # Print PCA components sweep table
    print("\n" + "=" * 60)
    print("PCA COMPONENTS SWEEP")
    print("=" * 60)
    print(f"{'PCA Components':<15} | {'Mean LOGO Acc':<15} | {'Std Dev':<10}")
    print("-" * 60)
    for pres in pca_results:
        npc_str = "None (raw)" if pres["n_components"] is None else (f"{pres['n_components']} (capped)" if pres["n_components"] == 50 else str(pres["n_components"]))
        print(f"{npc_str:<15} | {pres['mean_acc']*100:.2f}%         | {pres['std_acc']*100:.2f}%")
    print("=" * 60 + "\n")
    
    # Select optimal PCA n_components
    # If two values are within 1% of each other, prefer None or the smaller n_components
    best_pca_res = max(pca_results, key=lambda x: x["mean_acc"])
    candidates_within_1pct = [pres for pres in pca_results if pres["mean_acc"] >= best_pca_res["mean_acc"] - 0.01]
    
    none_res = next((pres for pres in candidates_within_1pct if pres["n_components"] is None), None)
    if none_res is not None:
        selected_pca_res = none_res
    else:
        selected_pca_res = min(candidates_within_1pct, key=lambda x: x["n_components"] if x["n_components"] is not None else 999)
        
    print(f"Selected Optimal PCA Components: {selected_pca_res['n_components']} (Mean Acc: {selected_pca_res['mean_acc']*100:.2f}%)\n")
    
    # 4. Print Per-Session Accuracy Table
    print("\n" + "=" * 80)
    print(f"PER-SESSION ACCURACY TABLE (Model: {best_cand['name']}, PCA: {selected_pca_res['n_components']})")
    print("=" * 80)
    print(f"{'Session':<30} | {'Class':<8} | {'N images':<10} | {'Accuracy when held out':<22}")
    print("-" * 80)
    for f in selected_pca_res["folds"]:
        # Get the class of the session
        idx = np.where(groups == f["session"])[0][0]
        class_str = "real" if y[idx] == 0 else "screen"
        n_imgs = np.sum(groups == f["session"])
        print(f"{f['session']:<30} | {class_str:<8} | {n_imgs:<10} | {f['accuracy']*100:.2f}%")
    print("=" * 80 + "\n")
    
    # 5. Optimize probability threshold
    best_threshold = select_threshold(y, selected_pca_res["oof_probs"])
    
    # 6. Final training and artifact export
    save_artifacts(
        X_orig=X_orig,
        X_aug_list=X_aug_list,
        y=y,
        groups=groups,
        chosen_model_name=best_cand["name"],
        chosen_clf_class=best_cand["class"],
        chosen_clf_kwargs=best_cand["kwargs"],
        best_threshold=best_threshold,
        folds=selected_pca_res["folds"],
        optimal_n_components=selected_pca_res["n_components"],
        oof_probs=selected_pca_res["oof_probs"]
    )
    
    print("\nTraining workflow completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
