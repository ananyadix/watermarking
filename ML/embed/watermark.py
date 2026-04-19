import numpy as np
import hashlib

# TEXT → BINARY
def text_to_binary(text):
    return ''.join(format(ord(c), '08b') for c in text)

# HASH (fragile watermark)
def generate_hash(data):
    return hashlib.sha256(data.tobytes()).hexdigest()

# Embed in DWT coefficients
def embed_in_coeff(coeff, binary):
    flat = coeff.flatten()

    for i in range(min(len(binary), len(flat))):
        flat[i] = int(flat[i]) & ~1 | int(binary[i])

    return flat.reshape(coeff.shape)

# MAIN embedding
def embed_dual_watermark(LL, LH, HL, HH, image, roi_mask, roni_mask, text):

    # ROI → fragile watermark
    roi_pixels = image[roi_mask == 255]
    roi_hash = generate_hash(roi_pixels)

    fragile_data = text + "|" + roi_hash
    fragile_binary = text_to_binary(fragile_data)

    # RONI → robust watermark (simplified)
    robust_binary = text_to_binary("ROBUST_" + text)

    # Embed
    LL = embed_in_coeff(LL, fragile_binary)   # important area
    HH = embed_in_coeff(HH, robust_binary)    # less important

    return LL, LH, HL, HH