import numpy as np
import cv2

def segment_roi(LL):
    # ROI: top-left quadrant
    mask = np.zeros_like(LL, dtype=bool)
    h, w = LL.shape
    mask[:h//2, :w//2] = True
    return mask

def segment_roni(LL, roi_mask):
    # RONI: everything outside ROI
    mask = ~roi_mask
    return mask