import cv2

def save_image(img, path="output.png"):
    cv2.imwrite(path, img)
    return path