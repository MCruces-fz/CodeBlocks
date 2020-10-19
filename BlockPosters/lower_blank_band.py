"""
@author: Miguel Cruces
"""

from imageio import imread, imwrite
import os
from os.path import join as join_path
import numpy as np


# Directorio raíz del proyecto
ROOT_DIR = os.path.abspath("./")

for file in os.listdir(ROOT_DIR):
    if not file.endswith('.py'):
        image_path = join_path(ROOT_DIR, file)
        image = imread(image_path)
        if len(image.shape) != 3:
            continue
        new_size = image.shape[0]//10
        band = image[-new_size:, :, :].copy()
        band[:, :, 1] = band[:, :, 0]
        band[:, :, 2] = band[:, :, 0]
        imwrite(f"{file.split('.')[0]}_modified.jpg", np.concatenate((image, band), axis=0))

