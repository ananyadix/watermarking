import cv2

def get_roi_mask(image):
    _, mask = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return mask

def split_roi_roni(image):
    roi_mask = get_roi_mask(image)
    roni_mask = cv2.bitwise_not(roi_mask)
    return roi_mask, roni_mask