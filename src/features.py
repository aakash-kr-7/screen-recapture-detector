"""
This module performs hand-engineered classical feature extraction on images for the binary image classifier.
It extracts physically-grounded cues to distinguish real photos from screen or printout recaptures,
including frequency-domain periodicity (moiré detection), color/gamut stats, texture/sharpness,
glare/specular highlights, and edge/geometry alignment.
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern

# Parallel list of human-readable feature names for model interpretability and debugging.
FEATURE_NAMES = [
    "freq_peak_to_average",
    "freq_high_to_low_energy_ratio",
    "color_saturation_mean",
    "color_saturation_std",
    "color_clipped_pixel_ratio",
    "color_l_dynamic_range",
    "color_cast_a",
    "color_cast_b",
    "texture_laplacian_var",
    "texture_lbp_entropy",
    "glare_blob_count",
    "glare_avg_hardness",
    "geom_aligned_line_count",
    "geom_avg_line_len_rel"
]


def _extract_frequency_features(gray: np.ndarray) -> tuple[float, float]:
    """
    Extracts frequency-domain periodicity features to detect grid patterns (moiré).
    
    Resizes image to 512x512, computes the 2D FFT magnitude spectrum, and checks
    for abnormal high-frequency peaks and energy distribution relative to low frequencies.
    """
    gray_512 = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
    
    # Compute 2D FFT and shift the zero-frequency component to the center
    f = np.fft.fft2(gray_512)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    
    # Generate coordinates and distances from center (256, 256)
    y, x = np.ogrid[:512, :512]
    r = np.sqrt((y - 256)**2 + (x - 256)**2)
    
    # Masks for low-frequency annulus (excluding DC spike at 0) and high-frequency annulus
    low_mask = (r >= 5) & (r < 50)
    high_mask = (r >= 60) & (r < 250)
    
    high_mag = magnitude_spectrum[high_mask]
    low_mag = magnitude_spectrum[low_mask]
    
    if len(high_mag) == 0 or len(low_mag) == 0:
        return 0.0, 0.0
        
    mean_high = np.mean(high_mag)
    peak_to_average = np.max(high_mag) / (mean_high + 1e-8)
    
    high_to_low_energy = np.sum(high_mag) / (np.sum(low_mag) + 1e-8)
    
    return float(peak_to_average), float(high_to_low_energy)


def _extract_color_features(img_bgr: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """
    Extracts color and gamut statistics from HSV and Lab spaces.
    
    Computes saturation mean/std, ratio of near-clipped pixels, L-channel dynamic range,
    and average color cast deviations.
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    
    S = img_hsv[:, :, 1]
    L = img_lab[:, :, 0]
    a = img_lab[:, :, 1]
    b = img_lab[:, :, 2]
    
    # Saturation statistics
    sat_mean = float(np.mean(S))
    sat_std = float(np.std(S))
    
    # Clipping statistics (luminance < 5 or > 250 in 8-bit scale)
    clipped_ratio = float(np.mean((L < 5) | (L > 250)))
    
    # Dynamic range with percentile clipping (99th - 1st) to ignore outliers
    l_min_pct = np.percentile(L, 1)
    l_max_pct = np.percentile(L, 99)
    l_dynamic_range = float(l_max_pct - l_min_pct)
    
    # White point color cast (deviation of a and b channels from neutral gray 128)
    cast_a = float(np.mean(a) - 128.0)
    cast_b = float(np.mean(b) - 128.0)
    
    return sat_mean, sat_std, clipped_ratio, l_dynamic_range, cast_a, cast_b


def _extract_texture_features(gray: np.ndarray) -> tuple[float, float]:
    """
    Extracts micro-texture statistics including Laplacian variance and LBP entropy.
    """
    h, w = gray.shape
    max_dim = 1024
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        gray_resized = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        gray_resized = gray.copy()

    # Laplacian variance as a proxy for sharpness
    lap_var = float(cv2.Laplacian(gray_resized, cv2.CV_64F).var())
    
    # Local Binary Pattern (LBP) entropy for regular micro-textures (pixel/paper grids)
    lbp = local_binary_pattern(gray_resized, P=8, R=1, method='uniform')
    
    # Uniform LBP with P=8 produces at most 10 distinct values
    n_bins = 10
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    # Shannon entropy
    hist = hist[hist > 0]
    lbp_entropy = float(-np.sum(hist * np.log2(hist)))
    
    return lap_var, lbp_entropy


def _extract_glare_features(gray: np.ndarray) -> tuple[float, float]:
    """
    Detects glare/specular highlights (small, high-contrast bright blobs)
    and computes count and average edge sharpness (hardness) of the blobs.
    """
    # Threshold for extremely bright regions
    bright_mask = (gray > 240).astype(np.uint8)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright_mask)
    
    # Sobel gradient magnitude to measure hardness/edge sharpness
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    valid_count = 0
    hardness_vals = []
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Filter for typical size of small specular reflections (e.g. 4 to 1000 pixels)
        if 4 <= area <= 1000:
            left = stats[i, cv2.CC_STAT_LEFT]
            top = stats[i, cv2.CC_STAT_TOP]
            w_b = stats[i, cv2.CC_STAT_WIDTH]
            h_b = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Crop region with 1 pixel padding for dilation boundary
            r_top = max(0, top - 1)
            r_bottom = min(gray.shape[0], top + h_b + 1)
            r_left = max(0, left - 1)
            r_right = min(gray.shape[1], left + w_b + 1)
            
            cropped_labels = labels[r_top:r_bottom, r_left:r_right]
            cropped_grad = grad_mag[r_top:r_bottom, r_left:r_right]
            
            blob_mask = (cropped_labels == i).astype(np.uint8)
            # Dilate mask and subtract original to get the boundary ring
            dilated = cv2.dilate(blob_mask, np.ones((3, 3), np.uint8))
            boundary = dilated - blob_mask
            
            if np.any(boundary > 0):
                # Hardness is the average gradient magnitude along the boundary
                hardness = float(np.mean(cropped_grad[boundary > 0]))
                hardness_vals.append(hardness)
                valid_count += 1
                
    avg_hardness = float(np.mean(hardness_vals)) if valid_count > 0 else 0.0
    
    return float(valid_count), avg_hardness


def _extract_edge_features(gray: np.ndarray) -> tuple[float, float]:
    """
    Finds straight axis-aligned lines as proxy indicators for screen frames or print edges.
    """
    h, w = gray.shape
    max_dim = 1000
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w = int(w * scale)
        new_h = int(h * scale)
        gray_resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        gray_resized = gray.copy()
        
    h_r, w_r = gray_resized.shape
    diag = np.sqrt(h_r**2 + w_r**2)
    
    # Run Canny edge detector
    edges = cv2.Canny(gray_resized, 50, 150)
    
    # Use optimized parameters for detecting long straight lines
    # Thresh=40, MinLen=8% of diagonal, Gap=10
    min_line_len = int(0.08 * diag)
    
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=40, 
                            minLineLength=min_line_len, maxLineGap=10)
    
    aligned_count = 0
    len_ratios = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            if length < 1e-8:
                continue
            
            # Map line angle in degrees to [0, 90]
            angle = np.abs(np.arctan2(dy, dx) * 180.0 / np.pi) % 90.0
            
            # Axis-aligned if close to horizontal (0 deg) or vertical (90 deg)
            if angle < 5.0 or angle > 85.0:
                aligned_count += 1
                len_ratios.append(length / diag)
                
    avg_len_ratio = float(np.mean(len_ratios)) if aligned_count > 0 else 0.0
    
    return float(aligned_count), avg_len_ratio


def extract_features(image_path: str) -> np.ndarray:
    """
    Loads an image from image_path, processes it using classical feature extraction,
    and returns a 14-dimensional feature vector.
    
    Raises a ValueError if the image is unreadable or corrupt.
    """
    if not os.path.exists(image_path):
        raise ValueError(f"Image path does not exist: '{image_path}'")
        
    try:
        # Load image via PIL to support a wide range of formats robustly
        with Image.open(image_path) as pil_img:
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            img_rgb = np.array(pil_img)
            
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        raise ValueError(f"Corrupt or unreadable image at '{image_path}': {str(e)}")
        
    # Extract feature values from each group helper
    f_freq1, f_freq2 = _extract_frequency_features(gray)
    f_col1, f_col2, f_col3, f_col4, f_col5, f_col6 = _extract_color_features(img_bgr)
    f_tex1, f_tex2 = _extract_texture_features(gray)
    f_glare1, f_glare2 = _extract_glare_features(gray)
    f_edge1, f_edge2 = _extract_edge_features(gray)
    
    # Construct the final feature vector
    features = np.array([
        f_freq1, f_freq2,
        f_col1, f_col2, f_col3, f_col4, f_col5, f_col6,
        f_tex1, f_tex2,
        f_glare1, f_glare2,
        f_edge1, f_edge2
    ], dtype=np.float32)
    
    return features


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/features.py <path_to_image>")
        sys.exit(1)
        
    img_path = sys.argv[1]
    print(f"Extracting features from: {img_path}")
    try:
        feats = extract_features(img_path)
        print("\nExtracted Features:")
        print("-" * 50)
        print(f"{'Feature Name':<30} | {'Value':<15}")
        print("-" * 50)
        for name, val in zip(FEATURE_NAMES, feats):
            print(f"{name:<30} | {val:<15.6f}")
        print("-" * 50)
        print(f"Total features extracted: {len(feats)}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
