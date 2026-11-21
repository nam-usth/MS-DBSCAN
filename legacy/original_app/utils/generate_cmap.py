#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Author       : HUYNH Vinh-Nam, MATTHEW Madany
# Email        : 
# Created Date : 14-November-2023
# Description  : 
"""
    A script that is responsible for creating color map 
    (source: Stackoverflow)

    This is the refined version to work with OpenCV
"""
#----------------------------------------------------------------------------

import cv2
import numpy as np
import random
from skimage.transform import resize

def distinguishable_colors(n, shuffle = True, sinusoidal = False, oscillate_tone = False): 
    ramp = ([1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]) if n > 3 else ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    
    coltrio = np.vstack(ramp)
    
    colmap = np.round(resize(coltrio, [n, 3], preserve_range = True, order = 1 if n > 3 else 3, mode = 'wrap'), 3)
    
    if sinusoidal: 
        colmap = np.sin(colmap * np.pi/2)
    
    colmap = [colmap[x,] for x  in range(colmap.shape[0])]
    
    if oscillate_tone:
        oscillate = [0, 1] * round(len(colmap) / 2 + 0.5)
        oscillate = [np.array([osc, osc, osc]) for osc in oscillate]
        colmap = [0.8 * colmap[x] + 0.2 * oscillate[x] for x in range(len(colmap))]
    
    #Whether to shuffle the output colors
    if shuffle:
        random.seed(1)
        random.shuffle(colmap)
        
    return colmap

# %% Main function

if __name__ == "__main__":
    # Define block size
    b_size = 25

    # Number of colors
    n_colors = 30

    # Create a blank image
    image_size = (b_size * n_colors, b_size)
    image = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255

    # Call cmap generator
    cmap = distinguishable_colors(n_colors, shuffle=False, sinusoidal=True, oscillate_tone=True)

    # Draw cmap to image
    for i in range(n_colors):
        start_point = (i * b_size, 0)
        end_point = ((i + 1) * b_size, b_size)
        image = cv2.rectangle(image, start_point, end_point, cmap[i] * 255, thickness=cv2.FILLED)

    # Display the image
    cv2.imshow('Generated cmap', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()