#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Author       : HUYNH Vinh-Nam
# Email        : huynh-vinh.nam@usth.edu.vn
# Created Date : 30-January-2024
# Description  : 
"""
    Automatic compute precision, recall, f1-score for segmented result

    Dice Coefficient (F1 Score)
    Dice = 2 * Area of overlap / Total area
    
"""
#----------------------------------------------------------------------------
import csv

from compute_surface_area import extract_surface_area

import fast_simplification

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pyvista as pv

def hash(x, y, z):
    # Input a 3D coordinate of a point
    # Compute unique hash value for that point
    # --> Get the point ID later
    x = 2*x if x > 0 else (-2)*x -1
    y = 2*y if y > 0 else (-2)*y -1
    z = 2*z if z > 0 else (-2)*z -1
    maxxyz = max(x, y, z)
    hash = maxxyz**3 + (2 * maxxyz * z) + z
    if (maxxyz == z):
        hash += max(x, y)**2
    if (y >= x):
        hash += x + y
    else:
        hash += y
    return hash


def retrieve_data(gt_file, seg_file):
    # Decode
    gt_dict = {}
    seg_dict = {}
    
    return gt_dict, seg_dict


def compute_area(gt_mesh, seg_mesh, gt_total_surface_area):

    seg_cell = []
    
    for cid in range(0, seg_mesh.n_cells):
        seg_cell.append(np.round(seg_mesh.get_cell(cid).points, decimals=7))
        # seg_cell.append(seg_mesh.get_cell(cid).points)

    # Vectorize the hash function
    vectorized_hash = np.vectorize(hash)

    # Hash-encoded cell
    hash_seg_cell = []

    for sgc in seg_cell:
        temp = vectorized_hash(sgc[:, 0], sgc[:, 1], sgc[:, 2])
        try:
            hash_seg_cell.append(hash(temp[0], temp[1], temp[2]))
        except:
            continue

    # Compute Area of overlap
    collision, ncol = gt_mesh.collision(seg_mesh, cell_tolerance=1)

    scalars = np.zeros(collision.n_cells, dtype=bool)
    scalars[collision.field_data['ContactCells']] = True

    overlap = 0

    for idx in np.unique(collision.field_data['ContactCells']):
        current_cell = collision.get_cell(idx)

        # [NOTE]
        # If only 3 points stand within the seg_mesh --> True, else False
        # And if the cell is truly intersection --> Add its area to overlap
        #
        # To make this easier, we can compute the hash for each triangle cell. 
        # Consider a cell i-th with [P1, P2, P3] where P1(x1, y1, z1), P2(x2, y2, z2), P3(x3, y3, z3)
        # --> Find a tuple C of 3 floats number [Hash(P1), Hash(P2), Hash(P3)]
        #    --> Hash for that cell is Hash(C)

        gt_p = current_cell.points

        gt_array = vectorized_hash(gt_p[:, 0], gt_p[:, 1], gt_p[:, 2])

        hash_gt_cell = hash(gt_array[0], gt_array[1], gt_array[2])

        if hash_gt_cell in hash_seg_cell:
            overlap += current_cell.cast_to_unstructured_grid().compute_cell_sizes()["Area"][0]
        else:
            # Re-correct
            scalars[idx] = False
        
    # Compute Total area
    total = gt_mesh.area + seg_mesh.area

    # Weight 
    weight = overlap / gt_total_surface_area

    # Compute Dice Coefficient
    # This one is actually: DSC (Dice Similarity Co-efficient)
    f1_score = 2 * overlap / total

    return f1_score, weight, overlap, collision, scalars

# %% Main function

if __name__ == "__main__":

    SELECTION_MODE = ['test', 'benchmark']
    MODE = SELECTION_MODE[1]

    if MODE == 'test':
            
        """
            [TEST]
        """
        # Folder paths
        dataset_dir = 'D:/Working/MTAP_journal/evaluation_dataset/Chen-22/'

        # Compute for each pair of file
        gt_mesh = pv.read('D:/Working/MTAP_journal/evaluation_dataset/Chen-22/GT/cube_subdivide/cube_seg_06.obj')

        seg_mesh = pv.read('D:/Working/MTAP_journal/evaluation_dataset/Chen-22/segmented_results/cube_subdivide/cube_subdivide-[6]-MS_DBSCAN-0.16-3-2-1708315848.1498888.ply')

        # Area data path
        csv_file = os.path.join(dataset_dir, 'area_data.csv')    
        df = pd.read_csv(csv_file)

        # Mesh path    
        model = 'icosahedron_subdivide_latest'

        query_exp = "file_name == '" + model + ".obj'"
        row_index = df.query(query_exp).index[0]

        f1_score, weight, overlap, collision, scalars = compute_area(gt_mesh, seg_mesh, df.iloc[row_index, 1])

        print("F1 score: ", f1_score, "- W: ", weight)

        # Visualization
        pl = pv.Plotter()

        _ = pl.add_mesh(collision, scalars=scalars, show_scalar_bar=True, cmap='jet')

        _ = pl.add_mesh(gt_mesh, style='wireframe', color='red', show_edges=True)

        _ = pl.add_mesh(seg_mesh, style='wireframe', color='green', show_edges=True)

        pl.show()

    elif MODE == 'benchmark':
            
        """
            [REAL BENCHMARK]
        """
        # Folder paths
        dataset_dir = 'D:/Working/MTAP_journal/evaluation_dataset/Siggraph-06/'
        
        # Area data path
        csv_file = os.path.join(dataset_dir, 'area_data.csv')    
        df = pd.read_csv(csv_file)

        # Mesh path
        space_mode = 'multiple'
        dataset = 'brick'
        model = dataset + '_part06'

        query_exp = "file_name == '" + model + ".obj'"
        row_index = df.query(query_exp).index[0]

        gt_folder = os.path.join(dataset_dir, 'GT', dataset, model)
        segmented_results_folder = os.path.join(dataset_dir, 'segmented_results', space_mode, dataset, model)

        stats_list = []
        temp = []

        # Iterate through files in segmented_results folder
        for segmented_file in os.listdir(segmented_results_folder):
            total_f1_score = 0
            total_span_count = 0

            if segmented_file.endswith(".ply"):
                segmented_path = os.path.join(segmented_results_folder, segmented_file)

                # Read the segmented mesh
                segmented_mesh = pv.read(segmented_path)

                # Iterate through files in GT folder
                for gt_file in os.listdir(gt_folder):
                    if gt_file.endswith(".obj"):
                        gt_path = os.path.join(gt_folder, gt_file)

                        # Read the ground truth mesh
                        gt_mesh = pv.read(gt_path)

                        f1_score, weight, overlap, collision, scalars = compute_area(gt_mesh, segmented_mesh, df.iloc[row_index, 1])

                        if f1_score != 0:
                            total_f1_score += f1_score * weight
                            total_span_count += 1

                        temp.append([space_mode, segmented_file, extract_surface_area(gt_mesh), extract_surface_area(segmented_mesh), overlap, f1_score, weight])
                
                if total_span_count == 0:
                    avg_f1_score = 0.0
                else:
                    avg_f1_score = total_f1_score

                if avg_f1_score > 1:
                    avg_f1_score = 1

                stats_list.append(avg_f1_score)


        # CSV file path
        csv_file = "benchmark.csv"

        # Writing the list to the CSV file
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([space_mode, model, stats_list])
