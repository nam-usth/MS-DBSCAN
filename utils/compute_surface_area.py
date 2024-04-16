#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Author       : HUYNH Vinh-Nam
# Email        : huynh-vinh.nam@usth.edu.vn
# Created Date : 12-March-2024
# Description  : 
"""
    Compute surface area for a given mesh
"""
#----------------------------------------------------------------------------
import fast_simplification

import csv
import numpy as np
import os
from os import walk
import pyvista as pv


def extract_surface_area(pv_mesh):
    surface_area = 0

    for cid in range(0, pv_mesh.n_cells):
        surface_area += pv_mesh.get_cell(cid).cast_to_unstructured_grid().compute_cell_sizes()["Area"][0]

    return surface_area

# %% Main function

if __name__ == "__main__":

    fol_dir = 'D:/Working/MTAP_journal/evaluation_dataset/CVPR_2022/'

    with open('data.csv', 'a', newline='') as csvfile:

        writer = csv.writer(csvfile)

        for file in os.listdir(fol_dir):
            file_name = os.path.join(fol_dir, file)

            if file_name.endswith('.obj'):
        
                pv_mesh = pv.read(file_name)

                if "subdivide" in file_name:
                    pv_mesh = pv_mesh
                else:
                    pv_mesh = fast_simplification.simplify_mesh(pv_mesh, target_reduction=0.9)
                
                    pv_mesh = pv_mesh.smooth(n_iter=100)

                surface_area = extract_surface_area(pv_mesh)

                writer.writerow([f"Area of {file_name}: {surface_area}"])