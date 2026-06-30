"""
This script loads the trained model pipeline and runs prediction on a single target image.
It prints the predicted probability of the image being a screen or printout recapture.
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
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found. Please run training first.")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file '{metadata_path}' not found. Please run training first.")
            
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
        raise FileNotFoundError(f"Image file does not exist: '{image_path}'")
        
    # extract_features raises ValueError if unreadable/corrupt
    feats = extract_features(image_path)
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
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
