from .dwt import apply_dwt, apply_idwt
from .roi import split_roi_roni
from .watermark import embed_dual_watermark

def embed_pipeline(image, text):

    # 1. DWT
    LL, LH, HL, HH = apply_dwt(image)

    # 2. ROI / RONI
    roi_mask, roni_mask = split_roi_roni(image)

    # 3. Embed watermark
    LL, LH, HL, HH = embed_dual_watermark(
        LL, LH, HL, HH,
        image,
        roi_mask,
        roni_mask,
        text
    )

    # 4. Reconstruct image
    watermarked = apply_idwt(LL, LH, HL, HH)

    return watermarked