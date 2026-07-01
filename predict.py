"""
predict.py - Inference Entrypoint CLI

What it is:
  This script is the main entrypoint for predicting whether a given image is a screen/printout recapture.

Why I did what I did:
  I wanted a zero-dependency runtime prediction script. By saving our trained StandardScaler and RandomForest 
  classifier in a single sklearn Pipeline ('models/model.pkl'), this CLI can perform inference in under a 
  second without loading heavy frameworks like PyTorch or TensorFlow. This ensures the script is portable, 
  on-device, and incurs $0 cloud API cost.

Contract:
  Input: Path to an image file (jpg/jpeg/png) as the first argument.
  Output: Prints a single float recapture probability (0.0 to 1.0) to stdout.
"""

import os
import sys
import json
import joblib

# Find project root dynamically based on script location
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
if os.path.basename(script_dir) == "src":
    project_root = os.path.dirname(script_dir)
else:
    project_root = script_dir

# Add directories to sys.path to ensure correct imports
if project_root not in sys.path:
    sys.path.insert(0, project_root)
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from src.features import extract_features
except ImportError:
    from features import extract_features

# Global variable cache for model pipeline and metadata
_MODEL_PIPELINE = None
_MODEL_METADATA = None


def load_model_once():
    """Loads the model pipeline and metadata once at startup."""
    global _MODEL_PIPELINE, _MODEL_METADATA
    if _MODEL_PIPELINE is None:
        model_path = os.path.join(project_root, "models", "model.pkl")
        metadata_path = os.path.join(project_root, "models", "metadata.json")
        
        # Verify model pipeline file path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Error: Model not found at '{model_path}'. Run src/train.py first to generate models/model.pkl"
            )
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Error: Model metadata not found at '{metadata_path}'. Run src/train.py first to generate models/metadata.json"
            )
            
        # model.pkl contains the full Pipeline (StandardScaler -> PCA -> Classifier)
        _MODEL_PIPELINE = joblib.load(model_path)
        with open(metadata_path, 'r') as f:
            _MODEL_METADATA = json.load(f)


def predict_probability(image_path: str) -> float:
    """
    Computes the predicted probability of the image being a screen or printout recapture.
    
    Args:
        image_path (str): Path to target image.
        
    Returns:
        float: Probability value between 0 and 1.
    """
    load_model_once()
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: Image file does not exist: '{image_path}'")
        
    try:
        # extract_features must return exactly 39 dimensions for the pipeline to accept it
        feats = extract_features(image_path)
    except Exception as e:
        raise RuntimeError(f"Error: Feature extraction failed for '{image_path}': {str(e)}")
        
    feats = feats.reshape(1, -1)
    prob = _MODEL_PIPELINE.predict_proba(feats)[0, 1]
    return float(prob)


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>", file=sys.stderr)
        sys.exit(1)
        
    image_path = sys.argv[1]
    try:
        prob = predict_probability(image_path)
        # Print ONLY the probability value rounded to 4 decimal places on stdout
        print(f"{prob:.4f}")
    except Exception as e:
        print(f"{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
