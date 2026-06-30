"""
This script benchmarks the inference latency of the hybrid feature extraction pipeline
across a sample of 20 images from the dataset. It reports the mean, median,
and 95th percentile latency in milliseconds.
"""

import os
import sys
import time
import numpy as np
from src.features import extract_features

def main():
    # Gather a sample of 20 images from the dataset (10 real, 10 screen)
    real_dir = os.path.join("dataset", "real")
    screen_dir = os.path.join("dataset", "screen")
    
    image_paths = []
    
    # Collect up to 10 from real
    if os.path.exists(real_dir):
        for root, _, files in os.walk(real_dir):
            for f in sorted(files):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(root, f))
                    if len(image_paths) >= 10:
                        break
            if len(image_paths) >= 10:
                break
                
    # Collect up to 10 from screen
    screen_paths = []
    if os.path.exists(screen_dir):
        for root, _, files in os.walk(screen_dir):
            for f in sorted(files):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    screen_paths.append(os.path.join(root, f))
                    if len(screen_paths) >= 10:
                        break
            if len(screen_paths) >= 10:
                break
                
    image_paths.extend(screen_paths)
    
    if len(image_paths) == 0:
        print("Error: No images found in dataset/ for benchmarking.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Benchmarking with {len(image_paths)} images...")
    
    # Warmup the feature extractor (first run loads MobileNet weights)
    print("Warming up feature extractor (loading MobileNet model to cache if not already)...")
    try:
        _ = extract_features(image_paths[0])
    except Exception as e:
        print(f"Warmup failed: {e}", file=sys.stderr)
        sys.exit(1)
        
    latencies = []
    for path in image_paths:
        try:
            t0 = time.perf_counter()
            _ = extract_features(path)
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0
            latencies.append(latency_ms)
            print(f"  {os.path.basename(path)}: {latency_ms:.2f} ms")
        except Exception as e:
            print(f"  Error benchmarking {os.path.basename(path)}: {e}")
            
    if not latencies:
        print("Error: All feature extractions failed during benchmarking.", file=sys.stderr)
        sys.exit(1)
        
    mean_lat = np.mean(latencies)
    median_lat = np.median(latencies)
    p95_lat = np.percentile(latencies, 95)
    
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    print(f"Mean Latency:   {mean_lat:.2f} ms")
    print(f"Median Latency: {median_lat:.2f} ms")
    print(f"95th Percentile:{p95_lat:.2f} ms")
    print("="*50)

if __name__ == "__main__":
    main()
