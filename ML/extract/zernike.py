# extract/zernike.py

import cv2
import numpy as np

def compute_rotation_invariant_zernike(image, radius=None, degree=8, block_size=64, threshold=0.1):
    """
    Compute block-wise Zernike-like moments to detect geometric attacks.
    Returns a 2D array of block features.
    
    Parameters:
    - image: 2D grayscale numpy array
    - radius: radius for Zernike computation (ignored in block approach)
    - degree: not used, placeholder for compatibility
    - block_size: size of blocks to divide the image
    - threshold: not used here, kept for compatibility
    """
    H, W = image.shape
    features = []

    # Ensure image is float in [0,1]
    img = image.astype(np.float32) / 255.0

    # Pad image to fit full blocks
    pad_h = (block_size - H % block_size) % block_size
    pad_w = (block_size - W % block_size) % block_size
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    H_pad, W_pad = img_padded.shape

    # Divide into blocks
    for i in range(0, H_pad, block_size):
        for j in range(0, W_pad, block_size):
            block = img_padded[i:i+block_size, j:j+block_size]
            # Compute simple rotation-invariant feature: mean + magnitude of FFT
            fft_block = np.fft.fft2(block)
            fft_mag = np.abs(fft_block)
            feature = np.array([np.mean(block), np.mean(fft_mag)])
            features.append(feature)

    return np.array(features)