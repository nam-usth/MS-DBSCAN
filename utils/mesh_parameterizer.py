#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Author       : HUYNH Vinh-Nam
# Email        : huynh-vinh.nam@usth.edu.vn
# Created Date : 31-August-2023
# Description  : 
"""
    A script that is responsible for doing UV unwrapping with various 
    methods (xatlas, blender)
"""
#----------------------------------------------------------------------------
import bpy
import cv2
import numpy as np
import open3d as o3d
import os
import pymeshlab
import trimesh
import xatlas

# %% Parametrize

def simple_xatlas_parameterize(tri_mesh, FILE_NAME):
    # The first way to parameterize a mesh
    # Export the parametrized mesh with xatlas (becareful of this operation)
    vmapping, indices, uvs = xatlas.parametrize(tri_mesh.vertices, tri_mesh.faces)
    xatlas.export(FILE_NAME + ".obj", tri_mesh.vertices[vmapping], indices, uvs)

    return vmapping, indices, uvs


def advanced_xatlas_parameterize(tri_mesh, FILE_NAME):
    # The second way to parameterize a mesh
    atlas = xatlas.Atlas()
    atlas.add_mesh(tri_mesh.vertices, tri_mesh.faces)
    
    chart_options = xatlas.ChartOptions()
    chart_options.max_iterations = 5

    pack_options = xatlas.PackOptions()
    pack_options.bruteForce = True
    pack_options.create_image = True

    atlas.generate(chart_options=chart_options, pack_options=pack_options)

    vmapping, indices, uvs = atlas[0]
    cv2.imwrite('Atlas.png', atlas.chart_image)

    # Export the parametrized mesh with xatlas (becareful of this operation)
    xatlas.export(FILE_NAME + ".obj", tri_mesh.vertices[vmapping], indices, uvs)

    return vmapping, indices, uvs


def blender_smart_UV_parameterize(INPUT_PATH, FILE_NAME):
    # Load the .obj file
    bpy.ops.import_scene.obj(filepath=INPUT_PATH)

    # Select the imported mesh object
    mesh_object = bpy.context.selected_objects[0]
    mesh_object.select_set(True)
    bpy.context.view_layer.objects.active = mesh_object

    # Switch to Edit mode
    bpy.ops.object.mode_set(mode='EDIT')

    # Select all vertices
    bpy.ops.mesh.select_all(action='SELECT')

    # Perform Smart UV Project
    # Angle unit: Radian
    bpy.ops.uv.smart_project(angle_limit=1.1519, margin_method='SCALED', island_margin=0.0, area_weight=0.0, correct_aspect=True, scale_to_bounds=False)

    # Unwrap the UVs using angle-based method
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.000)

    # Switch back to Object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Save the UV unwrapped mesh
    output_path = FILE_NAME + ".obj"
    bpy.ops.export_scene.obj(filepath=output_path, use_selection=True, use_materials=False)

    # Clean up the scene
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

# %% Main function

if __name__ == "__main__":
    blender_smart_UV_parameterize('../evaluation_dataset/SIGGRAPH_2006/cake_part03.obj', 'cp03')