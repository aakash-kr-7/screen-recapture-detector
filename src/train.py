"""
This script performs data loading, model validation, threshold selection,
final retraining, and model artifact exporting for the binary image classifier.
It trains a LogisticRegression classifier and a GradientBoostingClassifier,
compares them using Leave-One-Session-Out (LOGO) cross-validation,
determines the optimal decision threshold, and saves the final model pipeline.
"""

import os
import sys
import glob
import json
import datetime
import joblib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import feature extractor and standard names
from src.features import extract_features, FEATURE_NAMES

from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve


def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Scans dataset/real/ and dataset/screen/ for images grouped by sessions.
    Extracts features for each image, builds X, y, and groups arrays.
    
    Returns:
        X (np.ndarray): feature matrix of shape (N, 14)
        y (np.ndarray): labels of shape (N,) (0=real, 1=screen)
        groups (np.ndarray): session group names of shape (N,)
        paths (list[str]): original file paths
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
    
    # Print summary table and flag small sessions
    print("\n" + "=" * 80)
    print(f"{'Session Name':<35} | {'Class':<8} | {'Image Count':<12} | {'Status':<15}")
    print("=" * 80)
    for sess in sessions:
        label_str = "real" if sess["label"] == 0 else "screen"
        cnt = len(sess["images"])
        status = "OK"
        if cnt < 4:
            status = "FLAG (<4 images)"
        print(f"{sess['name']:<35} | {label_str:<8} | {cnt:<12} | {status:<15}")
    print("=" * 80 + "\n")
    
    # Extract features for all images
    X = []
    y = []
    groups = []
    paths = []
    
    total_images = sum(len(s["images"]) for s in sessions)
    processed = 0
    
    print(f"Extracting features from {total_images} images...")
    for sess in sessions:
        for img_path in sess["images"]:
            try:
                feats = extract_features(img_path)
                X.append(feats)
                y.append(sess["label"])
                groups.append(sess["name"])
                paths.append(img_path)
            except Exception as e:
                print(f"Warning: Skipping unreadable file '{img_path}' due to error: {e}")
            processed += 1
            if processed % 15 == 0 or processed == total_images:
                print(f"  Processed {processed}/{total_images} images...")
                
    return np.array(X), np.array(y), np.array(groups), paths


def run_validation(X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """
    Runs Leave-One-Session-Out Cross Validation on Logistic Regression and Gradient Boosting.
    Compares their performance, reports validation metrics, and picks the best model class.
    """
    logo = LeaveOneGroupOut()
    
    lr_folds = []
    gb_folds = []
    
    oof_probs_lr = np.zeros(len(y))
    oof_probs_gb = np.zeros(len(y))
    
    print("\nCommencing Leave-One-Session-Out (LOGO) Cross-Validation...")
    
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        held_out_session = groups[test_idx[0]]
        
        # 1. Logistic Regression Pipeline
        pipe_lr = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42))
        ])
        pipe_lr.fit(X_train, y_train)
        y_pred_lr = pipe_lr.predict(X_test)
        y_prob_lr = pipe_lr.predict_proba(X_test)[:, 1]
        oof_probs_lr[test_idx] = y_prob_lr
        
        lr_folds.append({
            "session": held_out_session,
            "accuracy": accuracy_score(y_test, y_pred_lr),
            "precision": precision_score(y_test, y_pred_lr, zero_division=0),
            "recall": recall_score(y_test, y_pred_lr, zero_division=0),
            "f1": f1_score(y_test, y_pred_lr, zero_division=0)
        })
        
        # 2. Gradient Boosting Pipeline
        pipe_gb = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(random_state=42))
        ])
        pipe_gb.fit(X_train, y_train)
        y_pred_gb = pipe_gb.predict(X_test)
        y_prob_gb = pipe_gb.predict_proba(X_test)[:, 1]
        oof_probs_gb[test_idx] = y_prob_gb
        
        gb_folds.append({
            "session": held_out_session,
            "accuracy": accuracy_score(y_test, y_pred_gb),
            "precision": precision_score(y_test, y_pred_gb, zero_division=0),
            "recall": recall_score(y_test, y_pred_gb, zero_division=0),
            "f1": f1_score(y_test, y_pred_gb, zero_division=0)
        })
        
    def report_metrics(folds, model_name):
        print(f"\n--- {model_name} LOGO Validation ---")
        print(f"{'Held-out Session':<35} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
        print("-" * 72)
        accs, precs, recs, f1s = [], [], [], []
        for f in folds:
            print(f"{f['session']:<35} | {f['accuracy']:.4f} | {f['precision']:.4f} | {f['recall']:.4f} | {f['f1']:.4f}")
            accs.append(f['accuracy'])
            precs.append(f['precision'])
            recs.append(f['recall'])
            f1s.append(f['f1'])
        print("-" * 72)
        print(f"{'MEAN':<35} | {np.mean(accs):.4f} | {np.mean(precs):.4f} | {np.mean(recs):.4f} | {np.mean(f1s):.4f}")
        print(f"{'STD':<35} | {np.std(accs):.4f} | {np.std(precs):.4f} | {np.std(recs):.4f} | {np.std(f1s):.4f}")
        print(f"Leave-one-session-out accuracy: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
        
    report_metrics(lr_folds, "Logistic Regression")
    report_metrics(gb_folds, "Gradient Boosting Classifier")
    
    # Calculate LOGO mean accuracies
    mean_acc_lr = np.mean([f['accuracy'] for f in lr_folds])
    mean_acc_gb = np.mean([f['accuracy'] for f in gb_folds])
    
    # Pick the best classifier
    if mean_acc_lr >= mean_acc_gb:
        chosen_model = "LogisticRegression"
        chosen_folds = lr_folds
        chosen_oof_probs = oof_probs_lr
        chosen_clf_class = LogisticRegression
        chosen_clf_kwargs = {"class_weight": "balanced", "max_iter": 2000, "random_state": 42}
        print(f"\nChosen best model: LogisticRegression (LOGO accuracy {mean_acc_lr*100:.2f}% vs GB {mean_acc_gb*100:.2f}%)")
    else:
        chosen_model = "GradientBoostingClassifier"
        chosen_folds = gb_folds
        chosen_oof_probs = oof_probs_gb
        chosen_clf_class = GradientBoostingClassifier
        chosen_clf_kwargs = {"random_state": 42}
        print(f"\nChosen best model: GradientBoostingClassifier (LOGO accuracy {mean_acc_gb*100:.2f}% vs LR {mean_acc_lr*100:.2f}%)")
        
    # Naive random 80/20 split comparison
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    pipe_rand = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', chosen_clf_class(**chosen_clf_kwargs))
    ])
    pipe_rand.fit(X_train, y_train)
    rand_preds = pipe_rand.predict(X_test)
    rand_acc = accuracy_score(y_test, rand_preds)
    
    print(f"\nRandom 80/20 split accuracy: {rand_acc*100:.2f}%")
    print("  (optimistic, not representative of generalization to new sessions — for reference only)")
    
    return chosen_model, chosen_folds, chosen_oof_probs, chosen_clf_class, chosen_clf_kwargs


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
    idx_05 = np.argmin(np.abs(thresholds - 0.5))
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


def save_artifacts(X: np.ndarray, y: np.ndarray, chosen_model_name: str, 
                   chosen_clf_class, chosen_clf_kwargs: dict, 
                   best_threshold: float, folds: list[dict]):
    """
    Retrains the chosen model class on the entire dataset and exports files to models/.
    """
    # Build final pipeline
    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', chosen_clf_class(**chosen_clf_kwargs))
    ])
    final_pipeline.fit(X, y)
    
    os.makedirs("models", exist_ok=True)
    
    # Save trained pipeline (includes scaled mapping and model coefficients)
    model_path = os.path.join("models", "model.pkl")
    joblib.dump(final_pipeline, model_path)
    print(f"Saved final trained model pipeline to '{model_path}'")
    
    # Compute LOGO accuracies
    logo_accs = [f['accuracy'] for f in folds]
    mean_logo_acc = float(np.mean(logo_accs))
    std_logo_acc = float(np.std(logo_accs))
    
    # Save metadata
    metadata = {
        "feature_names": FEATURE_NAMES,
        "chosen_threshold": best_threshold,
        "logo_mean_accuracy": mean_logo_acc,
        "logo_std_accuracy": std_logo_acc,
        "model_type": chosen_model_name,
        "training_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_image_count": int(len(y))
    }
    
    metadata_path = os.path.join("models", "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved model configuration metadata to '{metadata_path}'")
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    clf_obj = final_pipeline.named_steps['clf']
    
    if chosen_model_name == "LogisticRegression":
        importances = clf_obj.coef_[0]
        title = "Logistic Regression Feature Coefficients (Signed)"
        xlabel = "Coefficient Weight (Positive = Screen Recapture, Negative = Real Photo)"
        sorted_indices = np.argsort(importances)
    else:
        importances = clf_obj.feature_importances_
        title = "Gradient Boosting Feature Importances"
        xlabel = "Importance Score"
        sorted_indices = np.argsort(importances)
        
    y_pos = np.arange(len(FEATURE_NAMES))
    
    colors = ['skyblue' if i >= 0 else 'salmon' for i in importances[sorted_indices]] if chosen_model_name == "LogisticRegression" else 'skyblue'
    plt.barh(y_pos, importances[sorted_indices], align='center', alpha=0.8, color=colors)
    plt.yticks(y_pos, [FEATURE_NAMES[idx] for idx in sorted_indices])
    plt.xlabel(xlabel)
    plt.title(title)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8) if chosen_model_name == "LogisticRegression" else None
    plt.tight_layout()
    
    plot_path = os.path.join("models", "feature_importance.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved feature importance visualization to '{plot_path}'")


def main():
    print("=" * 80)
    print("BINARY IMAGE CLASSIFIER TRAINING WORKFLOW")
    print("=" * 80)
    
    # 1. Load data
    try:
        X, y, groups, _ = load_dataset()
    except Exception as exc:
        print(f"Error during data loading/feature extraction: {exc}", file=sys.stderr)
        sys.exit(1)
        
    if len(y) == 0:
        print("Error: No images found or feature extraction failed for all images.", file=sys.stderr)
        sys.exit(1)
        
    print(f"\nDataset loaded. Total instances: {len(y)}, Real count: {np.sum(y == 0)}, Screen count: {np.sum(y == 1)}")
    
    # 2. Run group cross validation
    chosen_model, folds, oof_probs, chosen_clf_class, chosen_clf_kwargs = run_validation(X, y, groups)
    
    # 3. Optimize probability threshold
    best_threshold = select_threshold(y, oof_probs)
    
    # 4. Final training and artifact export
    save_artifacts(X, y, chosen_model, chosen_clf_class, chosen_clf_kwargs, best_threshold, folds)
    
    print("\nTraining workflow completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
