#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Author       : HUYNH Vinh-Nam
# Email        : huynh-vinh.nam@usth.edu.vn
# Created Date : 31-August-2023
# Description  : 
"""
    A script that is responsible for exchanging 3D mesh format across 
    multiple libraries (Open3D, Meshlab, Trimesh, Vedo)
"""
#----------------------------------------------------------------------------
import numpy as np
import open3d as o3d
import os
import pymeshlab
import pyvista as pv
import trimesh
import vedo

# %% Convert mesh from open3d to trimesh and vice versa, open3d to pymeshlab, open3d to vedo, pyvista to trimesh and vice versa

def convert_open3d_to_trimesh_mesh(o3d_mesh):
    tri_mesh = trimesh.Trimesh(np.asarray(o3d_mesh.vertices), np.asarray(o3d_mesh.triangles), 
                               vertex_normals=np.asarray(o3d_mesh.vertex_normals))   
    return tri_mesh


def convert_trimesh_to_open3d_mesh(tri_mesh):
    # Convert vertices and faces from trimesh to open3d format
    vertices = o3d.utility.Vector3dVector(tri_mesh.vertices)
    faces = o3d.utility.Vector3iVector(tri_mesh.faces)
    
    # Create an open3d TriangleMesh object
    o3d_mesh = o3d.geometry.TriangleMesh(vertices, faces)
    
    return o3d_mesh


def convert_open3d_to_pymeshlab_mesh(o3d_mesh):
    verts = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    
    return pymeshlab.Mesh(verts, faces)


def convert_open3d_to_vedo_mesh(o3d_mesh):
    verts = []
    faces = []
    
    verts_list = np.asarray(o3d_mesh.vertices).tolist()
    faces_list = np.asarray(o3d_mesh.triangles).tolist()
    
    for i in range(0, len(verts_list)):
        verts.append(tuple(verts_list[i]))
        
    for i in range(0, len(faces_list)):
        faces.append(tuple(faces_list[i]))
    
    return vedo.Mesh([verts, faces])


def convert_pyvista_to_trimesh_mesh(pv_mesh):
    # Convert pyvista PolyData to trimesh Trimesh object
    faces_as_array = pv_mesh.faces.reshape((pv_mesh.n_faces, 4))[:, 1:] 
    tri_mesh = trimesh.Trimesh(pv_mesh.points, faces_as_array)
    return tri_mesh


def convert_trimesh_mesh_to_pyvista(tri_mesh):
    pv_mesh = pv.PolyData(np.array(tri_mesh.vertices), np.array(tri_mesh.faces))
    return pv_mesh


def convert_pyvista_to_pymeshlab_mesh(pv_mesh):
    verts = np.asarray(pv_mesh.points)
    faces = pv_mesh.faces.reshape((pv_mesh.n_faces, 4))[:, 1:] 

    return pymeshlab.Mesh(verts, faces)