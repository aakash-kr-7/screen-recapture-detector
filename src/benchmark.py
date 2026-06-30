"""
This script benchmarks the inference latency of the hybrid feature extraction
and classification pipeline using the core predict_probability function.
"""

import os
import sys
import time
import platform
import numpy as np

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
    from src.predict import predict_probability
except ImportError:
    from predict import predict_probability


def main():
    # Gather a sample of 20 images from the dataset (10 real, 10 screen)
    real_dir = os.path.join(project_root, "dataset", "real")
    screen_dir = os.path.join(project_root, "dataset", "screen")
    
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
        
    print(f"Benchmarking inference latency on {len(image_paths)} images...")
    
    # Warmup predict_probability (loads model, caches MobileNet model)
    print("Warming up model weights and feature extraction...")
    try:
        _ = predict_probability(image_paths[0])
    except Exception as e:
        print(f"Warmup failed: {e}", file=sys.stderr)
        sys.exit(1)
        
    latencies = []
    for path in image_paths:
        try:
            t0 = time.perf_counter()
            _ = predict_probability(path)
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0
            latencies.append(latency_ms)
            print(f"  {os.path.basename(path)}: {latency_ms:.2f} ms")
        except Exception as e:
            print(f"  Error predicting for {os.path.basename(path)}: {e}", file=sys.stderr)
            
    if not latencies:
        print("Error: All predictions failed.", file=sys.stderr)
        sys.exit(1)
        
    mean_lat = np.mean(latencies)
    median_lat = np.median(latencies)
    p95_lat = np.percentile(latencies, 95)
    
    print("\n" + "=" * 50)
    print("HARDWARE & CPU INFO")
    print("=" * 50)
    print(f"System:    {platform.system()} ({platform.release()})")
    print(f"Machine:   {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python:    {platform.python_version()}")
    print("=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Mean Latency:   {mean_lat:.2f} ms")
    print(f"Median Latency: {median_lat:.2f} ms")
    print(f"95th Percentile:{p95_lat:.2f} ms")
    print("=" * 50)


if __name__ == "__main__":
    main()
