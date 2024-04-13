#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Author       : HUYNH Vinh-Nam
# Email        : huynh-vinh.nam@usth.edu.vn
# Created Date : 03-September-2023
# Description  : 
"""
    Automatic compute all set of n-ring neighbors for a given mesh
"""
#----------------------------------------------------------------------------
import networkx as nx
import numpy as np
import pymeshlab
import pyvista as pv
import time
import trimesh


class GraphWrapper:
    def __init__(self, graph, vertices):
        self.graph = graph
        self.vertices = vertices


def get_n_ring_neighbor_vertices(graph, n, only_edge):
    n_ring_neighborhoods = []

    if n < 0:
        return n_ring_neighborhoods
    
    for vertex_index in range(len(graph.vertices)):
        n_cutoff = nx.single_source_shortest_path_length(graph.graph, vertex_index, cutoff=n)
        n_cutoff = list(n_cutoff.keys())

        if n == 0:
            n_minus_1_cutoff = []
        else:
            n_cutoff.remove(vertex_index)
            n_minus_1_cutoff = nx.single_source_shortest_path_length(graph.graph, vertex_index, cutoff=n-1)
            n_minus_1_cutoff = list(n_minus_1_cutoff.keys())
            n_minus_1_cutoff.remove(vertex_index)

        if only_edge:
            n_ring_neighborhoods.append(list(set(n_cutoff).difference(n_minus_1_cutoff)))
        else:
            n_ring_neighborhoods.append(list(set(n_cutoff)))

    return n_ring_neighborhoods


def get_n_ring_neighbor_vertices_with_trimesh(tri_mesh, n, only_edge):
    g_new = nx.Graph()
    g_new.add_nodes_from(tri_mesh.vertices)
    g_new.add_edges_from(tri_mesh.edges)
    
    return get_n_ring_neighbor_vertices(graph=GraphWrapper(g_new, tri_mesh.vertices), n=n, only_edge=only_edge)


def get_n_ring_neighbor_vertices_with_pymeshlab(mesh, n, only_edge):
    tri_mesh = trimesh.Trimesh(vertices=mesh.vertex_matrix(), faces=mesh.face_matrix())
    g_new = nx.Graph()
    g_new.add_nodes_from(tri_mesh.vertices)
    g_new.add_edges_from(tri_mesh.edges)

    return get_n_ring_neighbor_vertices(graph=GraphWrapper(g_new, tri_mesh.vertices), n=n, only_edge=only_edge)


def get_n_ring_neighbor_vertices_with_pyvista(pv_mesh, n, only_edge):
    vertices = pv_mesh.points
    faces = pv_mesh.faces.reshape(-1, 4)[:, 1:]
    edges = set(tuple(sorted([face[i], face[(i + 1) % len(face)]])) for face in faces for i in range(len(face)))
    
    g_new = nx.Graph()
    g_new.add_nodes_from(range(len(vertices)))
    g_new.add_edges_from(edges)

    return get_n_ring_neighbor_vertices(graph=GraphWrapper(g_new, vertices), n=n, only_edge=only_edge)


def n_ring_neighbor_visualizer(tri_mesh, n, vertex_idx):
    
    print(f'[ RUN      ] --- COMPUTE ALL {n}-RINGS ---')

    # Compute all n-rings for a given mesh 
    # (with mesh data as Trimesh.Mesh)
    start = time.time()
    nth = get_n_ring_neighbor_vertices_with_trimesh(tri_mesh, n, False)
    end = time.time()

    print(f'[     DONE ] COMPUTATION finished in {end - start} seconds for {len(tri_mesh.vertices)} vertices')

    # Draw it onto the scene
    p_original = trimesh.PointCloud([tri_mesh.vertices[vertex_idx].view(np.ndarray)], [200, 0, 0, 250])
    p_n_ring = trimesh.PointCloud([tri_mesh.vertices[v].view(np.ndarray) for v in nth[vertex_idx]], [0, 200, 0, 250])

    trimesh.Scene([tri_mesh, p_original, p_n_ring]).show(flags={'wireframe': True})


# %% Main function

# NOTE: This is the unit test for computing n-ring neighborhoods

if __name__ == "__main__":
    # '''
    #     Trimesh
    # '''
    # # Load a very simple mesh
    # m = trimesh.load('./evaluation_dataset/Siggraph-06/brick_part01_simplified.obj', force='mesh')

    # # Call visualizer function
    # # In this example, we draw the 3-ring neighbor for the 43rd vertex 
    # n_ring_neighbor_visualizer(m, 3, 43)

    
    '''
        PyVista
    '''
    # Load a very simple mesh
    pv_mesh = pv.read('./evaluation_dataset/Siggraph-06/brick_part01_simplified.obj')

    # Call visualizer function
    # In this example, we draw the 3-ring neighbor for the 43rd vertex 
    pid = 2890 # 2029
    
    nth_1 = get_n_ring_neighbor_vertices_with_pyvista(pv_mesh, 1, pid)
    nth_2 = get_n_ring_neighbor_vertices_with_pyvista(pv_mesh, 2, pid)
    nth_3 = get_n_ring_neighbor_vertices_with_pyvista(pv_mesh, 3, pid)

    # print(nth[pid])

    pl = pv.Plotter(shape=(1, 1))

    pl.add_mesh(pv_mesh, color="white")
    pl.add_mesh(pv_mesh, color="black", style="wireframe")

    pl.add_mesh(pv_mesh.points[pid], color="#FF0000", point_size=5)
    pl.add_mesh(pv_mesh.points[nth_1[pid]], color="#C254ED", point_size=4)
    pl.add_mesh(pv_mesh.points[nth_2[pid]], color="#FFC90E", point_size=4)
    pl.add_mesh(pv_mesh.points[nth_3[pid]], color="#75F94D", point_size=4)

    pl.show()