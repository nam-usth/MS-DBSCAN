#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Author       : HUYNH Vinh-Nam
# Email        : huynh-vinh.nam@usth.edu.vn
# Created Date : 22-November-2023
# Description  : 
"""
    Automatic compute all local resolution for each vertex of a given mesh
"""
#----------------------------------------------------------------------------
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

def local_point_cloud_resolution(points, k_nearest_neighbors):
    # Create KDTree using the points
    kdtree = KDTree(points)

    resolutions = []

    for i, point in enumerate(points):
        # Query k-nearest neighbors
        _, indices = kdtree.query(point, k=k_nearest_neighbors + 1) # +1 to exclude the point itself

        # Get the neighbors excluding the point itself
        neighbors = points[indices[1:]]

        # Calculate distances between every point in the neighbors list and all remaining neighbors
        distances = cdist(neighbors, neighbors)

        # Compute the mean and standard deviation of minimum distances
        mu_i = np.mean(distances)
        sigma_i = np.std(distances)

        # Estimate the local point cloud resolution
        beta_i = mu_i + 2 * sigma_i
        resolutions.append(beta_i)

    return resolutions


# %% Main function

if __name__ == "__main__":
    # Generate random 3D points
    num_points = 100
    points_cloud = np.random.rand(num_points, 3)

    # Set the number of nearest neighbors
    k = 5

    # Compute local point cloud resolution for the points
    res = local_point_cloud_resolution(points_cloud, k)

    # Display the computed resolutions for each point
    for i, beta_i in enumerate(res):
        print(f'Point {i}: Local Point Cloud Resolution = {beta_i}')

    print('============')

    # Compute average resolution of the whole point cloud data
    print(f'Average resolution: {np.mean(res)}')