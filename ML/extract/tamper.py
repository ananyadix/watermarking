import numpy as np

def compute_roi_hash(LL, roi_mask):
    # Simple sum as hash
    return int(np.sum(LL[roi_mask]))