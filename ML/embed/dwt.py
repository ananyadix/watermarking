import pywt

def apply_dwt(image):
    coeffs = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs
    return LL, LH, HL, HH

def apply_idwt(LL, LH, HL, HH):
    return pywt.idwt2((LL, (LH, HL, HH)), 'haar')