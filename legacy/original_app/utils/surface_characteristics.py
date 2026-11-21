#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Author       : HUYNH Vinh-Nam
# Email        : huynh-vinh.nam@usth.edu.vn
# Created Date : 31-October-2023
# Description  : 
"""
    Automatic compute the surface sharpness and roughness for a given mesh.
"""
#----------------------------------------------------------------------------
import numpy as np
import os
import pymeshlab
import pyvista as pv

from ring_neighbors_extractor import get_n_ring_neighbor_vertices_with_pymeshlab

# %% Compute mesh surface characteristics

def compute_surface_scalars(ms, r):
    # Compute curvature
    # [-] MeshLab filter name: "Discrete Curvatures"
    ms.compute_scalar_by_discrete_curvature_per_vertex(curvaturetype=1)
    H = ms.current_mesh().vertex_scalar_array()

    ms.compute_scalar_by_discrete_curvature_per_vertex(curvaturetype=2)
    G = ms.current_mesh().vertex_scalar_array()

    # Since:
    #   H = (k1 + k2)/2
    #   G = k1 * k2
    # then:
    #   (k1 - k2)^2 = 4H^2 - 4G

    V = 1/2 - 3/16 * H * r
    VD = 1 - (4*H**2 - 4*G) * 1/28 * (r**2)

    return V, VD


def compute_surface_characteristics(ms):
    sharpness, roughness = [], []

    # [OBJECTIVE 1]: Compute surface sharpness (multi-scale characteristics)
    N = 6 # or 7, 8 - depend

    # User pre-defined thresholds
    r_max = 0.1
    r_min = r_max / 2

    # Calculate r_i values as a vector
    r_i = np.delete(np.linspace(r_min, r_max, N), 0)

    # Volume descriptor
    V_desc = []
    for r in r_i:
        V_desc.append((compute_surface_scalars(ms, r)[0] - 1/2)**2)

    sharpness = np.sqrt(np.mean(np.array(V_desc), axis=0))

    # [OBJECTIVE 2]: Compute surface roughness (multi-scale characteristics)
    # Huynh V.N was using another way of implementation

    # NOTE: 
    # k = 1, threshold = 35
    # k = 2, threshold = 5
    # k = 3, threshold = 3
    # k = 4, threshold = 2

    k = 2

    mesh_normal_vec = ms.current_mesh().vertex_normal_matrix()
    mesh_vertices = ms.current_mesh().vertex_matrix()

    threshold = 5

    for idx, n_ring in enumerate(get_n_ring_neighbor_vertices_with_pymeshlab(ms.current_mesh(), k, True)):
        normal_diff = np.subtract(mesh_normal_vec[idx], mesh_normal_vec[n_ring])
        points_length = np.subtract(mesh_vertices[idx], mesh_vertices[n_ring])

        bending_energy = np.mean(np.divide(np.sum(normal_diff**2, axis=1), np.sum(points_length**2, axis=1)))

        # NOTE: 1 means original surface, 0 means fracture surface
        # roughness.append(1 if bending_energy < threshold else 0)

        roughness.append(bending_energy)

    return sharpness, roughness


# %% Main function

if __name__ == "__main__":
    FILE_NAME = './evaluation_dataset/SIGGRAPH_2006/brick_part02.obj'

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(FILE_NAME[:-4] + '.obj')

    V, VD = compute_surface_scalars(ms, 0.05)

    sharpness, roughness = compute_surface_characteristics(ms)

    # print(sharpness, roughness)

    # ms.current_mesh().add_vertex_custom_scalar_attribute(V, attribute_name='volume')
    # ms.current_mesh().add_vertex_custom_scalar_attribute(VD, attribute_name='volume distance')
    # ms.current_mesh().add_vertex_custom_scalar_attribute(sharpness, attribute_name='surface sharpness')
    # ms.current_mesh().add_vertex_custom_scalar_attribute(roughness, attribute_name='surface roughness')

    # save the current mesh without face color
    ms.save_current_mesh(FILE_NAME[:-4] + '_characterized.obj')

    # Read mesh
    pv_mesh = pv.read(FILE_NAME)
    pv_mesh["color_code"] = roughness 
    p = pv.Plotter()
    p.add_mesh(pv_mesh, scalars="color_code", cmap='jet')
    p.show()