"""
src/features.py - Classical Feature Extraction Module

What it is:
  This module implements the feature extraction engine for our detector. It processes an input image 
  and returns a 39-dimensional float32 vector representing 8 distinct classical image processing groups.

Why I did what I did:
  I replaced the heavy neural network embeddings (MobileNet) completely with classical image processing 
  techniques. Recaptures display distinct physical cues (Moiré, double JPEG compression boundary gaps, 
  specular reflection, and halftone print grids) that can be modeled directly using 2D Fast Fourier 
  Transforms, Discrete Cosine Transforms, color space analyses, LBP textures, and Hough lines.
  
  This classical engineering approach has three key benefits:
  1. No black-box dependencies or neural model files to ship.
  2. The features map to explainable physical concepts, making debugging failure modes straightforward.
  3. Extremely lightweight CPU footprints that run on standard on-device platforms without a GPU.
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern

# Parallel list of exactly 39 strings matching the feature vector output
FEATURE_NAMES = [
    # GROUP 1: PER-CHANNEL FFT PERIODICITY
    "freq_r_hf_ratio",
    "freq_g_hf_ratio",
    "freq_b_hf_ratio",
    "freq_rg_peak_corr",
    "freq_gb_peak_corr",
    "freq_rb_peak_corr",
    
    # GROUP 2: DCT BLOCK ARTIFACT DETECTION
    "dct_h_boundary_ratio",
    "dct_v_boundary_ratio",
    "dct_periodicity_score",
    "dct_block_mean_variance",
    "dct_gradient_direction_change",
    
    # GROUP 3: LOCAL SHARPNESS UNIFORMITY
    "sharp_mean_sharpness",
    "sharp_sharpness_std",
    "sharp_sharpness_cv",
    "sharp_center_to_edge_sharpness_ratio",
    "sharp_sharpness_range",
    
    # GROUP 4: LIGHTING-INVARIANT COLOR STATISTICS
    "color_b_channel_spatial_std",
    "color_a_channel_spatial_std",
    "color_sat_clipping_ratio",
    "color_sat_skewness",
    "color_luminance_spatial_std",
    "color_clipped_highlight_ratio",
    "color_sat_spatial_uniformity",
    
    # GROUP 5: NOISE ANALYSIS
    "noise_energy",
    "noise_uniformity",
    "noise_frequency_ratio",
    "noise_kurtosis",
    
    # GROUP 6: TEXTURE AND LBP
    "texture_laplacian_var",
    "texture_lbp_entropy",
    "texture_lbp_uniformity",
    
    # GROUP 7: EDGE AND GEOMETRY
    "geom_aligned_line_count",
    "geom_avg_line_len_rel",
    "geom_edge_density",
    "geom_edge_orientation_entropy",
    
    # GROUP 8: PRINTOUT DETECTION
    "print_halftone_mid_freq_ratio",
    "print_halftone_angular_regularity",
    "print_paper_mean_L",
    "print_paper_dark_ratio",
    "print_paper_noise_spatial_freq_peak"
]


# =====================================================================
# PRIVATE HELPER FUNCTIONS
# =====================================================================

def _extract_per_channel_fft(img_rgb: np.ndarray) -> list[float]:
    """
    Measures per-channel high-frequency energy ratio and cross-channel frequency peak correlation.
    
    Physical Signal:
        Screen subpixel RGB stripe patterns (both horizontal and vertical arrangements) 
        create correlated periodic peaks across R, G, B frequency spectra simultaneously. 
        Real scenes do not. Grayscale FFT averages channels and loses this cross-channel 
        correlation signal.
        
    Discriminative Power:
        Screen recaptures show correlated peaks in the frequency domain because the same subpixel
        grid creates peaks across all channels; real scenes show uncorrelated peak locations.
        
    Return Values:
        [r_hf_ratio, g_hf_ratio, b_hf_ratio, rg_peak_corr, gb_peak_corr, rb_peak_corr]
    """
    # Resize image to 512x512 for consistent frequency resolution
    img_512 = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_AREA) # 512: standardized frequency analysis resolution
    
    # Coordinates and distances from shifted center (256, 256)
    y, x = np.ogrid[:512, :512]
    r = np.sqrt((y - 256)**2 + (x - 256)**2) # 256: center frequency offset for 512x512 grid
    
    # Annulus masks
    low_mask = (r >= 5) & (r <= 63)  # 5 to 63: low-frequency band ignoring DC spike
    high_mask = (r >= 64) & (r <= 200) # 64 to 200: high-frequency band targeting subpixel moire periodicities
    
    hf_ratios = []
    channels_peaks = []
    
    # Get 2D coordinates of pixels in high-frequency annulus
    high_indices = np.argwhere(high_mask)
    
    for c in range(3): # Loop over R (0), G (1), B (2) channels
        channel = img_512[:, :, c]
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)
        
        # High and low frequency energy
        high_vals = mag[high_mask]
        low_vals = mag[low_mask]
        
        hf_ratio = float(np.sum(high_vals) / (np.sum(low_vals) + 1e-8)) # 1e-8: epsilon for stability
        hf_ratios.append(hf_ratio)
        
        # Get top-5 peak locations in high-frequency annulus
        top5_idx = np.argsort(high_vals)[-5:][::-1] # 5: number of peaks to track
        top5_coords = high_indices[top5_idx]
        channels_peaks.append(top5_coords)
        
    r_hf_ratio, g_hf_ratio, b_hf_ratio = hf_ratios
    
    # Compute cross-channel peak correlation for each pair (R-G, G-B, R-B)
    # R: 0, G: 1, B: 2
    pairs = [(0, 1), (1, 2), (0, 2)]
    peak_corrs = []
    
    for c1, c2 in pairs:
        peaks1 = channels_peaks[c1]
        peaks2 = channels_peaks[c2]
        
        matched = 0
        for p1 in peaks1:
            # Check if there is any peak in channels2 within 3 pixels
            dists = np.sqrt(np.sum((peaks2 - p1) ** 2, axis=1))
            if np.any(dists <= 3.0): # 3.0: pixel distance tolerance for correlated peaks
                matched += 1
        peak_corrs.append(float(matched / 5.0)) # 5.0: normalization for top-5 peaks
        
    rg_peak_corr, gb_peak_corr, rb_peak_corr = peak_corrs
    
    return [r_hf_ratio, g_hf_ratio, b_hf_ratio, rg_peak_corr, gb_peak_corr, rb_peak_corr]


def _extract_dct_block_artifacts(img_rgb: np.ndarray) -> list[float]:
    """
    Extracts features detecting JPEG double compression and grid boundary discontinuities.
    
    Physical Signal:
        When a screen displays a JPEG image and a camera re-photographs and saves it as 
        another JPEG, double quantization occurs. The two rounds of 8x8 DCT compression 
        create visible blocking artifacts at 8-pixel boundaries that single-capture real 
        photos never exhibit. This signal is completely lighting-independent.
        
    Discriminative Power:
        Recaptured photos exhibit periodic gradient spikes at 8-pixel intervals horizontally 
        and vertically from the double JPEG block grids, and higher energy in corresponding 
        2D DFT frequencies. Block means also show lower spatial variance compared to real photos 
        with natural high frequency/complex spatial content. Furthermore, double compression 
        artifacts create angular discontinuities at boundaries that natural textures do not have.
        
    Return Values:
        [h_boundary_ratio, v_boundary_ratio, dct_periodicity_score, block_mean_variance, 
         dct_gradient_direction_change]
    """
    # Convert to grayscale without resizing to keep pixel-aligned block grid intact
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # 1. Horizontal boundary ratio
    diff_h = np.abs(gray[:, 1:].astype(np.float64) - gray[:, :-1].astype(np.float64))
    # Columns of block boundaries: 8k-1 (0-indexed indices 7, 15, 23, ...)
    boundary_cols = [col for col in range(7, diff_h.shape[1]) if (col + 1) % 8 == 0] # 8: block size
    # Columns of block midpoints: 8k+3 (0-indexed indices 3, 11, 19, ...)
    midpoint_cols = [col for col in range(3, diff_h.shape[1]) if (col - 3) % 8 == 0] # 8: block size with offset 4
    
    if len(boundary_cols) > 0 and len(midpoint_cols) > 0:
        mean_boundary_h = np.mean(diff_h[:, boundary_cols])
        mean_midpoint_h = np.mean(diff_h[:, midpoint_cols])
        h_boundary_ratio = float(mean_boundary_h / (mean_midpoint_h + 1e-8)) # 1e-8: epsilon for stability
    else:
        h_boundary_ratio = 1.0 # default fallback
        
    # 2. Vertical boundary ratio
    diff_v = np.abs(gray[1:, :].astype(np.float64) - gray[:-1, :].astype(np.float64))
    # Rows of block boundaries: 8k-1 (0-indexed indices 7, 15, 23, ...)
    boundary_rows = [row for row in range(7, diff_v.shape[0]) if (row + 1) % 8 == 0] # 8: block size
    # Rows of block midpoints: 8k+3 (0-indexed indices 3, 11, 19, ...)
    midpoint_rows = [row for row in range(3, diff_v.shape[0]) if (row - 3) % 8 == 0] # 8: block size with offset 4
    
    if len(boundary_rows) > 0 and len(midpoint_rows) > 0:
        mean_boundary_v = np.mean(diff_v[boundary_rows, :])
        mean_midpoint_v = np.mean(diff_v[midpoint_rows, :])
        v_boundary_ratio = float(mean_boundary_v / (mean_midpoint_v + 1e-8)) # 1e-8: epsilon for stability
    else:
        v_boundary_ratio = 1.0 # default fallback

    # 3. 2D DFT of 16x16 patch from center
    h, w = gray.shape
    cy, cx = h // 2, w // 2 # Center coordinates
    # Extract 16x16 center patch
    patch = gray[max(0, cy-8):min(h, cy+8), max(0, cx-8):min(w, cx+8)].astype(np.float64) # 8: half of 16-pixel window
    if patch.shape != (16, 16): # 16: patch dimensions
        patch = cv2.resize(patch, (16, 16), interpolation=cv2.INTER_AREA)
        
    dft = np.fft.fft2(patch)
    dft_mag = np.abs(dft)
    
    # 8-pixel periodicity corresponds to 2 cycles in a 16x16 patch, i.e., frequency bin 2 (and symmetric bin 14)
    # Bins to query (horizontal, vertical, diagonal components)
    target_bins = [(2, 0), (14, 0), (0, 2), (0, 14), (2, 2), (2, 14), (14, 2), (14, 14)] # 2, 14: frequency bin index & symmetric alias
    bin2_energy = sum(dft_mag[u, v] for u, v in target_bins)
    total_ac_energy = np.sum(dft_mag) - dft_mag[0, 0] # 0, 0: DC center frequency
    
    dct_periodicity_score = float(bin2_energy / (total_ac_energy + 1e-8)) # 1e-8: epsilon for stability

    # 4. Variance of 8x8 block means
    H_blocks = h // 8 # 8: block height
    W_blocks = w // 8 # 8: block width
    if H_blocks > 0 and W_blocks > 0:
        cropped_gray = gray[:H_blocks * 8, :W_blocks * 8]
        # Reshape to (H_blocks, 8, W_blocks, 8) and compute block mean over spatial axes
        block_means = cropped_gray.reshape(H_blocks, 8, W_blocks, 8).mean(axis=(1, 3))
        block_mean_variance = float(np.var(block_means))
    else:
        block_mean_variance = 0.0
        
    # 5. Gradient direction change across 8x8 block boundaries
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) # 3: Sobel kernel size
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Horizontal boundaries
    h_boundaries = [y_b for y_b in range(8, h - 8) if y_b % 8 == 0] # 8: block boundary spacing
    h_diffs = []
    for y_b in h_boundaries:
        mean_dx_above = np.mean(sobel_x[y_b - 2 : y_b, :]) # 2: 2 rows above boundary
        mean_dy_above = np.mean(sobel_y[y_b - 2 : y_b, :])
        mean_dx_below = np.mean(sobel_x[y_b : y_b + 2, :]) # 2: 2 rows below boundary
        mean_dy_below = np.mean(sobel_y[y_b : y_b + 2, :])
        
        angle_above = np.arctan2(mean_dy_above, mean_dx_above)
        angle_below = np.arctan2(mean_dy_below, mean_dx_below)
        
        diff = np.abs(angle_above - angle_below)
        if diff > np.pi: # pi: wrap-around logic for angular difference
            diff = 2.0 * np.pi - diff
        h_diffs.append(diff)
    avg_h_diff = np.mean(h_diffs) if h_diffs else 0.0
    
    # Vertical boundaries
    v_boundaries = [x_b for x_b in range(8, w - 8) if x_b % 8 == 0] # 8: block boundary spacing
    v_diffs = []
    for x_b in v_boundaries:
        mean_dx_left = np.mean(sobel_x[:, x_b - 2 : x_b]) # 2: 2 columns left of boundary
        mean_dy_left = np.mean(sobel_y[:, x_b - 2 : x_b])
        mean_dx_right = np.mean(sobel_x[:, x_b : x_b + 2]) # 2: 2 columns right of boundary
        mean_dy_right = np.mean(sobel_y[:, x_b : x_b + 2])
        
        angle_left = np.arctan2(mean_dy_left, mean_dx_left)
        angle_right = np.arctan2(mean_dy_right, mean_dx_right)
        
        diff = np.abs(angle_left - angle_right)
        if diff > np.pi: # pi: wrap-around logic for angular difference
            diff = 2.0 * np.pi - diff
        v_diffs.append(diff)
    avg_v_diff = np.mean(v_diffs) if v_diffs else 0.0
    
    dct_gradient_direction_change = float((avg_h_diff + avg_v_diff) / 2.0) # 2.0: average horizontal & vertical components
        
    return [h_boundary_ratio, v_boundary_ratio, dct_periodicity_score, block_mean_variance, dct_gradient_direction_change]


def _extract_sharpness_uniformity(img_rgb: np.ndarray) -> list[float]:
    """
    Extracts spatial sharpness distribution features to measure depth of field and lens focus uniformity.
    
    Physical Signal:
        A flat screen is equidistant from the camera lens across its entire surface, 
        producing highly uniform sharpness throughout the image. Real 3D scenes have 
        natural depth variation causing uneven sharpness with falloff from the focal plane. 
        This signal is content-independent and lighting-independent.
        
    Discriminative Power:
        Real photos have higher variance in sharpness between spatial regions (high std/CV, 
        high range) and tend to be sharper in the center than at the edges. Screens have 
        uniform spatial sharpness (low CV, low range, center-to-edge ratio close to 1).
        
    Return Values:
        [mean_sharpness, sharpness_std, sharpness_cv, center_to_edge_sharpness_ratio, sharpness_range]
    """
    # Resize grayscale version to max dimension 1024 to speed up calculation without losing focus details
    h, w = img_rgb.shape[:2]
    max_dim = 1024 # 1024: standard resize target for sharpness/texture resolution
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_resized = img_rgb.copy()
        
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    
    # Divide the image into a 4x4 grid of 16 non-overlapping patches
    patch_h = H // 4 # 4: grid rows
    patch_w = W // 4 # 4: grid cols
    gray_cropped = gray[:patch_h * 4, :patch_w * 4]
    
    sharpness_grid = np.zeros((4, 4), dtype=np.float64)
    for r in range(4): # 4: rows
        for c in range(4): # 4: cols
            patch = gray_cropped[r * patch_h : (r + 1) * patch_h, c * patch_w : (c + 1) * patch_w]
            # Compute Laplacian variance for each patch as a sharpness measure
            sharpness_grid[r, c] = float(cv2.Laplacian(patch, cv2.CV_64F).var())
            
    mean_sharpness = float(np.mean(sharpness_grid))
    sharpness_std = float(np.std(sharpness_grid))
    sharpness_cv = float(sharpness_std / (mean_sharpness + 1e-8)) # 1e-8: epsilon for stability
    
    # Extract center and edge patch values
    # Center patches (inner 2x2): (1,1), (1,2), (2,1), (2,2)
    center_indices = [(1, 1), (1, 2), (2, 1), (2, 2)]
    # Edge patches (outer non-corner border): (0,1), (0,2), (1,0), (2,0), (3,1), (3,2), (1,3), (2,3)
    edge_indices = [(0, 1), (0, 2), (1, 0), (2, 0), (3, 1), (3, 2), (1, 3), (2, 3)]
    
    center_means = [sharpness_grid[r, c] for r, c in center_indices]
    edge_means = [sharpness_grid[r, c] for r, c in edge_indices]
    
    mean_center = np.mean(center_means)
    mean_edge = np.mean(edge_means)
    center_to_edge_sharpness_ratio = float(mean_center / (mean_edge + 1e-8)) # 1e-8: epsilon for stability
    
    max_sharp = np.max(sharpness_grid)
    min_sharp = np.min(sharpness_grid)
    sharpness_range = float((max_sharp - min_sharp) / (mean_sharpness + 1e-8)) # 1e-8: epsilon for stability
    
    return [mean_sharpness, sharpness_std, sharpness_cv, center_to_edge_sharpness_ratio, sharpness_range]


def _extract_color_statistics(img_rgb: np.ndarray) -> list[float]:
    """
    Extracts spatial and distribution-based color statistics invariant to global illumination.
    
    Physical Signal:
        Emissive display panels render content with highly uniform color temperature 
        and high brightness highlights. They also exhibit artificial saturation properties 
        due to backlighting and software color boosting.
        
    Discriminative Power:
        Screens show lower spatial variance in chromaticity (a and b channels) compared 
        to real scenes that have complex lighting/shadows. Screens also clip saturation 
        levels (sat > 220) and highlights (L > 250) more aggressively, and show right-skewed 
        saturation distributions from vivid graphics. Saturation is also spatially 
        non-uniform compared to bright real objects like windows/sky or paper printouts.
        
    Return Values:
        [b_channel_spatial_std, a_channel_spatial_std, sat_clipping_ratio, sat_skewness, 
         luminance_spatial_std, clipped_highlight_ratio, sat_spatial_uniformity]
    """
    H, W = img_rgb.shape[:2]
    patch_h = H // 3 # 3: grid rows
    patch_w = W // 3 # 3: grid cols
    
    # 1. CIELab spatial color statistics
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
    lab_cropped = lab[:patch_h * 3, :patch_w * 3]
    
    L_channel = lab_cropped[:, :, 0]
    a_channel = lab_cropped[:, :, 1]
    b_channel = lab_cropped[:, :, 2]
    
    a_means = []
    b_means = []
    L_means = []
    
    for r in range(3): # 3: rows
        for c in range(3): # 3: cols
            p_a = a_channel[r * patch_h : (r + 1) * patch_h, c * patch_w : (c + 1) * patch_w]
            p_b = b_channel[r * patch_h : (r + 1) * patch_h, c * patch_w : (c + 1) * patch_w]
            p_L = L_channel[r * patch_h : (r + 1) * patch_h, c * patch_w : (c + 1) * patch_w]
            
            a_means.append(np.mean(p_a))
            b_means.append(np.mean(p_b))
            L_means.append(np.mean(p_L))
            
    a_channel_spatial_std = float(np.std(a_means))
    b_channel_spatial_std = float(np.std(b_means))
    luminance_spatial_std = float(np.std(L_means))
    
    # 2. HSV saturation statistics
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    
    # 3. Saturation clipping statistics (exclude near-white/bright highlights and dark regions)
    L_full = lab[:, :, 0]
    # sat_clipping_ratio: saturation > 220 AND luminance L between 40 and 230
    sat_clipping_ratio = float(np.mean((sat > 220) & (L_full >= 40) & (L_full <= 230))) # 220: sat clip threshold, 40/230: luminance bounds
    
    # Compute saturation histogram with 8 bins (placeholder/unused in return but kept for consistency)
    _ = np.histogram(sat, bins=8, range=(0, 256)) # 8: number of bins, 256: 8-bit scale
    
    # sat_skewness: Skewness of saturation distribution
    sat_flat = sat.flatten().astype(np.float64)
    mean_sat = np.mean(sat_flat)
    std_sat = np.std(sat_flat)
    if std_sat < 1e-8: # 1e-8: check for flat saturation
        sat_skewness = 0.0
    else:
        sat_skewness = float(np.mean((sat_flat - mean_sat) ** 3) / (std_sat ** 3 + 1e-8)) # 1e-8: epsilon for stability
        
    # Highlight clipping statistics (fraction of pixels with L > 250 in Lab space)
    clipped_highlight_ratio = float(np.mean(L_full > 250)) # 250: highlight clipping limit close to peak brightness (255)
    
    # 4. Saturation spatial uniformity (coefficient of variation of patch saturation mean values)
    sat_patch_means = []
    # Crop sat to multiple of 3
    sat_cropped = sat[:patch_h * 3, :patch_w * 3]
    for r in range(3): # 3: rows
        for c in range(3): # 3: cols
            p_sat = sat_cropped[r * patch_h : (r + 1) * patch_h, c * patch_w : (c + 1) * patch_w]
            sat_patch_means.append(np.mean(p_sat))
    mean_sat_patches = np.mean(sat_patch_means)
    std_sat_patches = np.std(sat_patch_means)
    sat_spatial_uniformity = float(std_sat_patches / (mean_sat_patches + 1e-8)) # 1e-8: epsilon for stability
    
    return [b_channel_spatial_std, a_channel_spatial_std, sat_clipping_ratio, sat_skewness, 
            luminance_spatial_std, clipped_highlight_ratio, sat_spatial_uniformity]


def _extract_noise_features(img_rgb: np.ndarray) -> list[float]:
    """
    Extracts sensor noise residual features and analyzes their spatial and frequency distributions.
    
    Physical Signal:
        Camera sensor noise (PRNU — Photo Response Non-Uniformity) has specific spatial 
        characteristics. When photographing a screen, camera noise is layered on top of 
        the screen's own rendering/compression noise, producing a distinctly different 
        noise field than photographing a real scene directly.
        
    Discriminative Power:
        Screens show more spatially uniform noise residuals (lower coefficient of variation 
        of noise variances), structured frequency content (higher high-to-low ratio from 
        subpixel interference/compression grid), and heavier tails (higher kurtosis) from 
        double-compression artifacts.
        
    Return Values:
        [noise_energy, noise_uniformity, noise_frequency_ratio, noise_kurtosis]
    """
    # Convert to grayscale and resize to max dimension 512 for consistent noise evaluation resolution
    h, w = img_rgb.shape[:2]
    max_dim = 512 # 512: standardized scale for sensor noise analysis
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_resized = img_rgb.copy()
        
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    gray_double = gray.astype(np.float64)
    
    # Extract noise residual: apply Gaussian blur and subtract from original
    blurred = cv2.GaussianBlur(gray_double, (3, 3), 1.0) # 3: kernel size, 1.0: Gaussian sigma
    residual = gray_double - blurred
    
    # 1. noise_energy
    noise_energy = float(np.mean(residual ** 2))
    
    # 2. noise_uniformity (divided into 4x4 grid)
    H, W = residual.shape
    patch_h = H // 4 # 4: rows
    patch_w = W // 4 # 4: cols
    res_cropped = residual[:patch_h * 4, :patch_w * 4]
    
    variances = []
    for r in range(4): # 4: rows
        for c in range(4): # 4: cols
            patch = res_cropped[r * patch_h : (r + 1) * patch_h, c * patch_w : (c + 1) * patch_w]
            variances.append(np.var(patch))
            
    mean_var = np.mean(variances)
    std_var = np.std(variances)
    noise_uniformity = float(std_var / (mean_var + 1e-8)) # 1e-8: epsilon for stability
    
    # 3. noise_frequency_ratio (ratio of high-frequency outer annulus to low-frequency inner annulus)
    res_512 = cv2.resize(residual, (512, 512), interpolation=cv2.INTER_AREA) # 512: normalize grid for FFT
    f = np.fft.fft2(res_512)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    
    y, x = np.ogrid[:512, :512]
    r = np.sqrt((y - 256)**2 + (x - 256)**2) # 256: center frequency offset for 512x512 grid
    
    # Inner and outer frequency annulus masks
    low_mask = (r >= 5) & (r <= 63) # 5 to 63: low frequency band
    high_mask = (r >= 64) & (r <= 250) # 64 to 250: high frequency band targeting sensor noise + grid frequencies
    
    sum_low = np.sum(mag[low_mask])
    sum_high = np.sum(mag[high_mask])
    noise_frequency_ratio = float(sum_high / (sum_low + 1e-8)) # 1e-8: epsilon for stability
    
    # 4. noise_kurtosis (excess kurtosis of residual)
    flat_res = residual.flatten()
    mean_res = np.mean(flat_res)
    std_res = np.std(flat_res)
    if std_res < 1e-8: # 1e-8: check for zero variance noise
        noise_kurtosis = 0.0
    else:
        noise_kurtosis = float(np.mean((flat_res - mean_res) ** 4) / (std_res ** 4 + 1e-8) - 3.0) # 3.0: subtract 3 for excess kurtosis
        
    return [noise_energy, noise_uniformity, noise_frequency_ratio, noise_kurtosis]


def _extract_texture_features(img_rgb: np.ndarray) -> list[float]:
    """
    Extracts spatial micro-texture sharpness and Local Binary Pattern (LBP) regularity.
    
    Physical Signal:
        Real photos of three-dimensional scenes have complex focus structures and 
        high-variance edge frequencies, whereas recaptured displays are limited by 
        screen-panel resolution limits, pixel pitch, or printing halftone dots, 
        creating artificial spatial regularity and lower overall micro-texture entropy.
        
    Discriminative Power:
        Screens have lower Laplacian variance (less natural high frequency focus), lower 
        LBP entropy, and higher LBP uniformity (reflecting highly structured micro-texture grids).
        
    Return Values:
        [texture_laplacian_var, texture_lbp_entropy, texture_lbp_uniformity]
    """
    # Resize to max dimension 1024 for texture scaling consistency
    h, w = img_rgb.shape[:2]
    max_dim = 1024 # 1024: standard texturing analysis scale
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img_resized = cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        img_resized = img_rgb.copy()
        
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    
    # 1. Laplacian variance as a global sharpness proxy
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # 2. Local Binary Pattern (LBP) features (P=8, R=1, method='uniform')
    # P=8: 8 neighborhood points, R=1: radius 1 pixel
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    
    # Uniform LBP with P=8 produces 10 distinct bins
    n_bins = 10 # 10: standard uniform LBP histogram bins
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    
    # Normalize histogram to probabilities
    hist_norm = hist.astype(np.float64) / (np.sum(hist) + 1e-8) # 1e-8: epsilon for stability
    
    # Shannon entropy of LBP histogram
    lbp_entropy = float(-np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0] + 1e-15))) # 1e-15: avoid log of zero
    
    # LBP uniformity (sum of squared bin probabilities)
    lbp_uniformity = float(np.sum(hist_norm ** 2))
    
    return [laplacian_var, lbp_entropy, lbp_uniformity]


def _extract_geometry_features(img_rgb: np.ndarray) -> list[float]:
    """
    Extracts geometric line structure features, Canny edge density, and gradient orientation entropy.
    
    Physical Signal:
        Screen recaptures and paper printouts frequently feature distinct straight borders, 
        bezel frames, or page boundaries. They also contain flat content with highly aligned 
        horizontal and vertical edge distributions.
        
    Discriminative Power:
        Screens show a higher count of long axis-aligned lines (Hough), longer relative 
        average line lengths, characteristic edge density (UI elements vs natural scenes), 
        and lower gradient orientation entropy (concentrated along horizontal/vertical directions).
        
    Return Values:
        [geom_aligned_line_count, geom_avg_line_len_rel, edge_density, edge_orientation_entropy]
    """
    h, w = img_rgb.shape[:2]
    max_dim = 1000 # 1000: standard geometric/Hough transform scale
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_resized = img_rgb.copy()
        
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    h_r, w_r = gray.shape
    diag = np.sqrt(h_r**2 + w_r**2)
    
    # 1. Canny edge detection
    edges = cv2.Canny(gray, 50, 150) # 50: lower Canny threshold, 150: upper threshold
    
    # 2. edge_density
    edge_density = float(np.mean(edges > 0))
    
    # 3. Hough Line Transform for axis-aligned lines
    min_line_len = int(0.08 * diag) # 0.08: lines must be at least 8% of the image diagonal
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=40, # 1: radial, pi/180: angular resolution, 40: voting threshold
                            minLineLength=min_line_len, maxLineGap=10) # 10: maximum pixel gap to connect segments
    
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
                
            angle = np.abs(np.arctan2(dy, dx) * 180.0 / np.pi) % 90.0 # Map angle to [0, 90] range
            
            # Check axis alignment with a 5.0 degree tolerance around horizontal (0) and vertical (90)
            if angle < 5.0 or angle > 85.0: # 5.0, 85.0: horizontal/vertical thresholds
                aligned_count += 1
                len_ratios.append(length / diag)
                
    geom_aligned_line_count = float(aligned_count)
    geom_avg_line_len_rel = float(np.mean(len_ratios)) if aligned_count > 0 else 0.0
    
    # 4. Gradient orientation entropy
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) # 3: kernel size
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    angles_deg = (np.arctan2(sobel_y, sobel_x) * 180.0 / np.pi) % 180.0 # Map to [0, 180) range
    
    # Filter for significant gradients to ignore flat background noise
    grad_mask = magnitude > 10.0 # 10.0: gradient magnitude noise floor threshold
    
    if np.any(grad_mask):
        valid_angles = angles_deg[grad_mask]
        hist_orient, _ = np.histogram(valid_angles, bins=8, range=(0, 180)) # 8: orientation bins, 180: degrees
        hist_orient_norm = hist_orient.astype(np.float64) / (np.sum(hist_orient) + 1e-8)
        edge_orientation_entropy = float(-np.sum(hist_orient_norm[hist_orient_norm > 0] * np.log2(hist_orient_norm[hist_orient_norm > 0] + 1e-15))) # 1e-15: avoid log of zero
    else:
        edge_orientation_entropy = 0.0
        
    return [geom_aligned_line_count, geom_avg_line_len_rel, edge_density, edge_orientation_entropy]


def _extract_printout_features(img_rgb: np.ndarray) -> list[float]:
    """
    Detects physical artifacts specific to printed paper: halftone dot patterns 
    from inkjet/laser printing, paper grain in noise residual, reflective (not 
    emissive) luminance profile, and ink color gamut characteristics.
    
    Physical Signal:
        Printed images have high reflectance, narrow luminance ranges, fine-grained 
        halftone distributions, and structured dot frequency orientations.
        
    Discriminative Power:
        Distinguishes printed recaptures (s7_printout) from emissive screens and 
        real 3D photos. Printouts have a characteristic halftone FFT ratio, angular 
        regularity of dots (often at 45/75 degree angles), high paper mean L, 
        and low dark pixel counts.
        
    Return Values:
        [print_halftone_mid_freq_ratio, print_halftone_angular_regularity, 
         print_paper_mean_L, print_paper_dark_ratio, print_paper_noise_spatial_freq_peak]
    """
    # 1. Halftone FFT features
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_512 = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA) # 512: standardized size for FFT
    f = np.fft.fft2(gray_512)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    
    y, x = np.ogrid[:512, :512]
    r = np.sqrt((y - 256)**2 + (x - 256)**2) # 256: shifted center
    
    # Mid-high frequency annulus for halftones: radius 30-80
    mid_high_mask = (r >= 30) & (r <= 80) # 30 to 80: halftone frequency range
    annulus_energy = np.sum(mag[mid_high_mask])
    total_energy = np.sum(mag)
    halftone_mid_freq_ratio = float(annulus_energy / (total_energy + 1e-8)) # 1e-8: epsilon for stability
    
    # Halftone angular regularity: find top 10 peaks in the annulus, excluding horizontal/vertical axes by 20 degrees
    dy = y - 256
    dx = x - 256
    angles_full = np.abs(np.arctan2(dy, dx) * 180.0 / np.pi)
    angles_mod_90 = angles_full % 90.0
    non_axis_mask = (angles_mod_90 >= 20.0) & (angles_mod_90 <= 70.0) # 20.0 to 70.0: exclude 20 degrees near 0/90 axes
    
    valid_mask = mid_high_mask & non_axis_mask
    high_indices = np.argwhere(valid_mask)
    vals = mag[valid_mask]
    
    if len(vals) >= 10: # 10: number of peaks to track
        top10_idx = np.argsort(vals)[-10:][::-1]
        top10_coords = high_indices[top10_idx]
        angles = np.array([np.arctan2(p[0] - 256, p[1] - 256) for p in top10_coords]) # 256: shift center
        
        # Avoid symmetry cancellation by mapping to [0, pi) and doubling angles
        angles_mod = angles % np.pi
        angles_double = 2.0 * angles_mod
        
        sum_cos = np.sum(np.cos(angles_double))
        sum_sin = np.sum(np.sin(angles_double))
        R = np.sqrt(sum_cos**2 + sum_sin**2) / 10.0 # 10.0: sample size normalization
        halftone_angular_regularity = float(1.0 - R)
    else:
        halftone_angular_regularity = 1.0 # default fallback
        
    # 2. Paper luminance profile
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
    L_channel = lab[:, :, 0]
    paper_mean_L = float(np.mean(L_channel))
    paper_dark_ratio = float(np.mean(L_channel < 30)) # 30: threshold for dark paper shadow pixels
    
    # 3. Paper noise texture (dominant frequency in noise residual FFT)
    h, w = img_rgb.shape[:2]
    max_dim = 512 # 512: standardized scale for sensor noise analysis
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img_resized = cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        img_resized = img_rgb.copy()
        
    gray_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    gray_double = gray_resized.astype(np.float64)
    blurred = cv2.GaussianBlur(gray_double, (3, 3), 1.0) # 3: kernel size, 1.0: Gaussian sigma
    residual = gray_double - blurred
    
    res_512 = cv2.resize(residual, (512, 512), interpolation=cv2.INTER_AREA) # 512: normalize grid for FFT
    f_res = np.fft.fft2(res_512)
    fshift_res = np.fft.fftshift(f_res)
    mag_res = np.abs(fshift_res)
    
    y_r, x_r = np.ogrid[:512, :512]
    r_res = np.sqrt((y_r - 256)**2 + (x_r - 256)**2) # 256: shift center
    mask_res = r_res >= 5 # 5: exclude DC and low-frequency components
    
    valid_indices = np.argwhere(mask_res)
    vals_res = mag_res[mask_res]
    if len(vals_res) > 0:
        max_idx = np.argmax(vals_res)
        y_max, x_max = valid_indices[max_idx]
        dom_freq = np.sqrt((y_max - 256)**2 + (x_max - 256)**2) # 256: center shift
        paper_noise_spatial_freq_peak = float(dom_freq / 512.0) # 512.0: normalize by image size
    else:
        paper_noise_spatial_freq_peak = 0.0
        
    return [halftone_mid_freq_ratio, halftone_angular_regularity, paper_mean_L, paper_dark_ratio, paper_noise_spatial_freq_peak]


# =====================================================================
# PUBLIC INTERFACE
# =====================================================================

def extract_features(image_path) -> np.ndarray:
    """
    Loads an image from image_path (or accepts a PIL Image / numpy array directly),
    processes it using 8 classical image processing groups, and returns a 39-dimensional 
    flat float32 feature vector.
    
    Raises a ValueError if the image is unreadable or corrupt.
    """
    if isinstance(image_path, str):
        if not os.path.exists(image_path):
            raise ValueError(f"Image path does not exist: '{image_path}'")
            
        try:
            # Load image via PIL to support a wide range of formats robustly
            with Image.open(image_path) as pil_img:
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                img_rgb = np.array(pil_img)
        except Exception as e:
            raise ValueError(f"Corrupt or unreadable image at '{image_path}': {str(e)}")
    elif isinstance(image_path, Image.Image):
        if image_path.mode != "RGB":
            image_path = image_path.convert("RGB")
        img_rgb = np.array(image_path)
    elif isinstance(image_path, np.ndarray):
        img_rgb = image_path.copy()
    else:
        raise ValueError(f"Invalid input type: {type(image_path)}")
        
    try:
        # 1. Per-Channel FFT Periodicity (6 features)
        g1 = _extract_per_channel_fft(img_rgb)
        # 2. DCT Block Artifact Detection (5 features)
        g2 = _extract_dct_block_artifacts(img_rgb)
        # 3. Local Sharpness Uniformity (5 features)
        g3 = _extract_sharpness_uniformity(img_rgb)
        # 4. Lighting-Invariant Color Statistics (7 features)
        g4 = _extract_color_statistics(img_rgb)
        # 5. Noise Analysis (4 features)
        g5 = _extract_noise_features(img_rgb)
        # 6. Texture and LBP (3 features)
        g6 = _extract_texture_features(img_rgb)
        # 7. Edge and Geometry (4 features)
        g7 = _extract_geometry_features(img_rgb)
        # 8. Printout Detection (5 features)
        g8 = _extract_printout_features(img_rgb)
        
        # Concatenate all groups (6 + 5 + 5 + 7 + 4 + 3 + 4 + 5 = 39 features)
        features = np.concatenate([g1, g2, g3, g4, g5, g6, g7, g8]).astype(np.float32)
        
        return features
    except Exception as e:
        raise ValueError(f"Failed to extract features: {str(e)}")


# =====================================================================
# DEBUG ENTRYPOINT
# =====================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single Image: python src/features.py <path_to_image>")
        print("  Comparison:   python src/features.py <path_to_real_image> <path_to_screen_image>")
        sys.exit(1)
        
    if len(sys.argv) == 2:
        # Single image mode
        img_path = sys.argv[1]
        print(f"Extracting features from: {img_path}")
        try:
            feats = extract_features(img_path)
            print("\nExtracted Features:")
            print("-" * 55)
            print(f"{'Feature Name':<35} | {'Value':<15}")
            print("-" * 55)
            for name, val in zip(FEATURE_NAMES, feats):
                print(f"{name:<35} | {val:<15.6f}")
            print("-" * 55)
            print(f"Total features extracted: {len(feats)}")
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
            
    elif len(sys.argv) >= 3:
        # Comparison mode (Real vs Screen)
        img_path1 = sys.argv[1]
        img_path2 = sys.argv[2]
        print(f"Comparing Features:")
        print(f"  Real Image:   {img_path1}")
        print(f"  Screen Image: {img_path2}")
        try:
            feats1 = extract_features(img_path1)
            feats2 = extract_features(img_path2)
            
            print("\n" + "=" * 105)
            print(f"{'Feature Name':<35} | {'Real Value':<18} | {'Screen Value':<18} | {'Difference':<15}")
            print("=" * 105)
            for name, val1, val2 in zip(FEATURE_NAMES, feats1, feats2):
                diff = val2 - val1
                print(f"{name:<35} | {val1:<18.6f} | {val2:<18.6f} | {diff:<+15.6f}")
            print("=" * 105)
            print(f"Total features compared: {len(feats1)}")
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
