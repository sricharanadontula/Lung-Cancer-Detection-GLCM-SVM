# ==========================================
# Image Preprocessing
# ==========================================

import cv2

IMAGE_SIZE = 256

def preprocess_image(image_path):
    """
    Reads image, resizes and converts to grayscale
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Invalid image path")

    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray

