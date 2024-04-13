#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Author       : HUYNH Vinh-Nam
# Email        : huynh-vinh.nam@usth.edu.vn
# Created Date : 31-August-2023
# Description  : 
"""
    A script that reads a point-cloud/mesh file and makes it ready as
    3D mesh (Open3D) 
"""
#----------------------------------------------------------------------------
import numpy as np
import open3d as o3d
import os

# %% Point cloud

def read_pcd(FILE_NAME):
    # Try open and read the FILE_NAME
    try:
        pcd = o3d.io.read_point_cloud(FILE_NAME, format='xyzn')
        #pcd = pcd.voxel_down_sample(voxel_size=0.05) # 0.225, 0.075
        return pcd
    except:
        print("Couldn't read file")


def convert_pcd_to_mesh(pcd):
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=7) #depth=9
        return mesh
    except:
        print("Invalid Data...")


# %% Mesh

def read_mesh(FILE_NAME):
    try:
        mesh = o3d.io.read_triangle_mesh(FILE_NAME)
        return mesh
    except:
        print("Couldn't read file")


# %% Input data handler

def input_handler(FILE_NAME, mode="mesh"):
    if (mode == 'point_cloud'):
        data = convert_pcd_to_mesh(read_pcd(FILE_NAME))
        
    if (mode == 'mesh'):
        data = read_mesh(FILE_NAME)
    return data


# %% Output data handler

def output_handle(FILE_NAME, mesh):
    try:
        o3d.io.write_triangle_mesh(FILE_NAME, mesh)
    except:
        print("Couldn't write file")