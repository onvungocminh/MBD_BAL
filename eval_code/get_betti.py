import sys
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
import os
sys.path.append('Betti_Compute/')
import ext_libs.Gudhi as gdh
import cv2
import pdb

def betti_number(imagely):
    # imagely = np.array(Image.open(imagely))
    # print(input_path)
    # if input_path.split('.')[-1] == 'png':
    #     imagely = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    # else:
    #     imagely = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    #     imagely = (imagely > 0.5).astype(np.uint8)


    width,height = imagely.shape
    imagely[width - 1, :] = 0
    imagely[:, height - 1] = 0
    imagely[0, :] = 0
    imagely[:, 0] = 0
    temp = gdh.compute_persistence_diagram(imagely, i = 1)
    betti_number = len(temp)
    return betti_number
