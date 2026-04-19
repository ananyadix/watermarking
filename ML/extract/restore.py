import numpy as np

def extract_fragile_watermark(LL_water, roi_mask):
    fragile_bits = (LL_water[roi_mask] > np.mean(LL_water[roi_mask])).astype(int)
    return fragile_bits

def extract_robust_watermark(LL_water, roni_mask):
    return LL_water[roni_mask].copy()