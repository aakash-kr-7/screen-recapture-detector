"""
This script loads the trained model pipeline and runs prediction on a single target image.
It prints the predicted probability of the image being a screen or printout recapture.
"""

import os
import sys
import joblib
from src.features import extract_features

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image path does not exist: '{image_path}'", file=sys.stderr)
        sys.exit(1)
        
    # Load the trained model pipeline
    model_path = os.path.join("models", "model.pkl")
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please run training first.", file=sys.stderr)
        sys.exit(1)
        
    try:
        pipeline = joblib.load(model_path)
    except Exception as e:
        print(f"Error: Failed to load model pipeline: {e}", file=sys.stderr)
        sys.exit(1)
        
    try:
        # Extract the hybrid 590-dimensional feature vector
        feats = extract_features(image_path)
        # Reshape for sklearn input (1, n_features)
        feats = feats.reshape(1, -1)
        # Get class probabilities
        prob = pipeline.predict_proba(feats)[0, 1]
        print(f"{prob:.6f}")
    except Exception as e:
        print(f"Error: Feature extraction or prediction failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
