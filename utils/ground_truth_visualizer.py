#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Author       : HUYNH Vinh-Nam
# Email        : huynh-vinh.nam@usth.edu.vn
# Created Date : 21-March-2024
# Description  : 
"""
    A script that is responsible for reading 3D surfaces on GT folder 
    and displaying them with color code
"""
#----------------------------------------------------------------------------
import numpy as np
import os
import pyvista as pv


def ground_truth_sub_meshes_combiner(gt_folder):
    gt_sub_meshes = []
    counter = 0

    for file in os.listdir(gt_folder):
        if file.endswith(".ply") or file.endswith(".obj"):
            sub_mesh = pv.read(os.path.join(gt_folder, file))
            sub_mesh["cell"] = counter * np.ones(sub_mesh.n_cells)
            gt_sub_meshes.append(sub_mesh)
            counter += 1

    gt_mesh = pv.MultiBlock(gt_sub_meshes)

    return gt_mesh, counter


# %% Main function

if __name__ == "__main__":
    
    # dataset_dir = 'D:/Working/MTAP_journal/evaluation_dataset/Siggraph-06/'
    dataset_dir = 'D:/Working/MTAP_journal/evaluation_dataset/Chen-22/'

    # Mesh path
    space_mode = ['single', 'multiple']
    dataset = 'cake'
    model = dataset + '_part01'
    model = 'pyramid_subdivide'

    gt_folder = os.path.join(dataset_dir, 'GT', dataset, model)
    gt_folder = os.path.join(dataset_dir, 'GT', model)
    seg_single_folder = os.path.join(dataset_dir, 'segmented_results', space_mode[0], dataset, model)
    seg_multiple_folder = os.path.join(dataset_dir, 'segmented_results', space_mode[1], dataset, model)

    gt_mesh, counter = ground_truth_sub_meshes_combiner(gt_folder)
    # seg_mesh_single, _ = ground_truth_sub_meshes_combiner(seg_single_folder)
    # seg_mesh_multiple, _ = ground_truth_sub_meshes_combiner(seg_multiple_folder)

    # counter = 0

    # Visualization
    # pl = pv.Plotter(shape=(1, 3))
    pl = pv.Plotter(shape=(1, counter+1))

    pl.subplot(0, 0)

    # pl.camera_position = [
    #         (0.01, 0.02, 0.03),
    #         (0.1, 0, -0.1),
    #         (-0.01, 0.02, -0.02),
    #     ]
    
    _ = pl.add_mesh(gt_mesh) #, cmap='jet')

    for i in range(1, counter+1):
        pl.subplot(0, i)
        _ = pl.add_mesh(gt_mesh[i-1]) #, style='wireframe')

    # pl.subplot(0, 1)
    # _ = pl.add_mesh(seg_mesh_single, scalars="cell", cmap='jet')

    # pl.subplot(0, 2)
    # _ = pl.add_mesh(seg_mesh_multiple, scalars="cell", cmap='jet')

    pl.show()