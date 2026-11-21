#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Author       : HUYNH Vinh-Nam
# Email        : huynh-vinh.nam@usth.edu.vn
# Created Date : 02-September-2023
# Description  : 
"""
    A script that is responsible for computing curvature and exporting
    curvature texture of a given mesh (Meshlab)
"""
#----------------------------------------------------------------------------

import numpy as np
import os
import pymeshlab
import trimesh

# %% Compute mesh curvature and save the curvature information as a texture

def compute_mesh_curvature(FILE_NAME):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(FILE_NAME[:-4] + '.obj')

    # [QUESTION]: Should we do simplification here or do it before the UV parameterization? (Some faces can be concatenated into one segment)
    # Simplify mesh before apply curvature filter (to avoid crashing)
    # The mesh's vertex_number should be smaller than 40,000
    # [-] MeshLab filter name: "Simplification: Quadric Edge Collapse Decimation (with texture)"
    while (ms.current_mesh().vertex_number() > 40000):
        ms.meshing_decimation_quadric_edge_collapse_with_texture(targetperc=0.9, qualitythr=0.3, extratcoordw=1, preservenormal=True, planarquadric=True)

    # Compute curvature
    # [-] MeshLab filter name: "Compute curvature principal directions"
    curv = ms.compute_curvature_principal_directions_per_vertex(method=3, curvcolormethod=0, autoclean=True)

    # [-] MeshLab filter name: "Discrete Curvatures"
    #curv = ms.compute_scalar_by_discrete_curvature_per_vertex(curvaturetype=1)
    mesh_scalar = ms.current_mesh().vertex_scalar_array()

    print(mesh_scalar, np.max(mesh_scalar), np.min(mesh_scalar))

    # Normalize curvature
    # Normalized within the range [0, 1] - Built-in function
    # The built-in function only set min=0.0 and max=1.0
    #ms.compute_scalar_by_function_per_vertex(q='q', normalize=True)

    ms.compute_texmap_from_color(textname='K_' + FILE_NAME[:-4] + '_additional_info.png', pullpush=False)
    ms.compute_texcoord_transfer_wedge_to_vertex()
    ms.save_current_mesh(file_name='temp.ply', save_vertex_quality=True, save_vertex_coord=True, save_wedge_texcoord=True)