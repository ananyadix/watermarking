# extract/extract.py

import cv2
import numpy as np
from .dwt_utils import apply_dwt, apply_idwt
from .roi_utils import segment_roi, segment_roni
from .tamper import compute_roi_hash
from .zernike import compute_rotation_invariant_zernike
from .metrics import calculate_mse, calculate_psnr
from skimage.metrics import structural_similarity as ssim

# -----------------------------
# Fragile Watermark Extraction
# -----------------------------
def extract_fragile_watermark(LL_water, roi_mask):
    fragile_bits = (LL_water[roi_mask] > np.mean(LL_water[roi_mask])).astype(int)
    return fragile_bits

# -----------------------------
# Robust Watermark Extraction
# -----------------------------
def extract_robust_watermark(LL_water, roni_mask):
    return LL_water[roni_mask].copy()

# -----------------------------
# Extraction, Verification & Restoration
# -----------------------------
def extract_and_restore_numpy(orig_image, water_image):
    """
    Extract watermark, check integrity & geometric attacks, and restore.
    Inputs:
        orig_image: numpy array (grayscale)
        water_image: numpy array (grayscale)
    Returns:
        geo: geometric verification result
        integrity: fragile watermark integrity result
        restored: restored image (numpy array)
        psnr, mse, ssim_score: image quality metrics
    """

    # --- Metrics ---
    mse_val = calculate_mse(orig_image, water_image)
    psnr_val = calculate_psnr(orig_image, water_image)
    ssim_score, _ = ssim(orig_image, water_image, full=True)

    # --- DWT ---
    LL_o, (LH_o, HL_o, HH_o) = apply_dwt(orig_image)
    LL_w, (LH_w, HL_w, HH_w) = apply_dwt(water_image)

    # --- ROI / RONI ---
    roi_mask = segment_roi(LL_o)
    roni_mask = segment_roni(LL_o, roi_mask)

    # --- Fragile Watermark Extraction & Tamper Detection ---
    fragile_bits = extract_fragile_watermark(LL_w, roi_mask)
    orig_hash = compute_roi_hash(LL_o, roi_mask)
    diff = np.mean(np.abs(LL_o[roi_mask] - LL_w[roi_mask]))
    integrity = "Authentic" if diff < 3 else "Tampered"

    # --- Robust Watermark Extraction & Geometric Verification ---
    robust_data = extract_robust_watermark(LL_w, roni_mask)
    zernike_orig = compute_rotation_invariant_zernike(orig_image)
    zernike_water = compute_rotation_invariant_zernike(water_image)

    # Compare block features
    diff = np.mean(np.abs(zernike_orig - zernike_water), axis=1)  # mean per block
    attack_blocks = np.sum(diff > 0.1)  # threshold per block

    geo = "Safe" if attack_blocks < len(diff) * 0.1 else "Geometric Attack Detected"
    # If more than 10% blocks differ significantly, report attack

    # --- Reversible Restoration ---
    LL_restored = LL_w.copy()
    LL_restored[roni_mask] = robust_data
    restored = apply_idwt((LL_restored, (LH_w, HL_w, HH_w)))

    return geo, integrity, restored, psnr_val, mse_val, ssim_score