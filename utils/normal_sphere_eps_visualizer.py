#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Author       : HUYNH Vinh-Nam
# Email        : huynh-vinh.nam@usth.edu.vn
# Created Date : 15-March-2024
# Description  : 
"""
    Visualize the normal epsilon area on the Normal Domain
    
"""
#----------------------------------------------------------------------------

import numpy as np
import os
import pyvista as pv


# %% Callback implementation

def callback(point):
    global old_intersect

    try:
        _ = p.remove_actor(old_intersect)
    except:
        pass

    # Create a sphere centered at the picked point
    # Threshold radius is set to eps_normal
    sphere_A = pv.Sphere(center=point, radius=0.16)

    # Compute the intersection between two spheres
    intersection, _, _ = sphere.intersection(sphere_A)

    # Add newly created intersect
    current_intersect = p.add_mesh(intersection, color="red", line_width=2)

    # Set the current intersect to be the old one and will be deleted in 
    # the next turn
    old_intersect = current_intersect


# %% Main function

if __name__ == "__main__":

    dataset_dir = 'D:/Working/MTAP_journal/evaluation_dataset/SIGGRAPH_2006/'

    # Mesh path
    space_mode = 'multiple'
    dataset = 'cake'
    model = dataset + '_part01'

    pv_mesh = pv.read(os.path.join(dataset_dir, 'segmented_results', space_mode, dataset, model, 'cake_part01-[0]-MS_DBSCAN-0.16-3-2-1709172405.031267.ply'))

    normals = pv_mesh.point_normals

    # Create the first sphere
    sphere = pv.Sphere(center=([0, 0, 0]), radius=1)
    
    # Visualize the result
    p = pv.Plotter()
    p.add_mesh(normals)
    p.add_mesh(sphere, show_edges=True, opacity=0.025, pickable=True)
    p.enable_point_picking(callback=callback)
    p.show()