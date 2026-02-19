# ==========================================
# Advanced GLCM Feature Extraction
# ==========================================

import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_glcm_features(gray_image):
    """
    Extract multi-angle, multi-distance GLCM features
    """

    distances = [1, 2]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    glcm = graycomatrix(
        gray_image,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True
    )

    features = []

    for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
        values = graycoprops(glcm, prop)
        features.extend(values.flatten())

    return np.array(features)
