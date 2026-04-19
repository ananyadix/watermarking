import pywt
import numpy as np

def apply_dwt(image):
    # Single-level DWT using Haar
    coeffs2 = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs2
    return LL, (LH, HL, HH)

def apply_idwt(coeffs):
    LL, (LH, HL, HH) = coeffs
    reconstructed = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    return reconstructed