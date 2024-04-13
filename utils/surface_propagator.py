#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Author       : HUYNH Vinh-Nam, NGUYEN Hoang-Ha
# Email        : huynh-vinh.nam@usth.edu.vn
# Created Date : 12-September-2023
# Description  : 
"""
    Automatic compute all set of surfaces for a given mesh.

    The idea is to extract 3D poly boundary. Then from the
    some initialized seed vertices, the algorithm will
    try to propagate until reaching the vertex on boundary.

    This works features networkx and maybe CUDA in the 
    future (for HPC).
"""
#----------------------------------------------------------------------------
from collections import Counter

import fast_simplification

import hdbscan

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import os
import pymeshlab
import pyvista as pv
from pyvista import examples
from pyvista import _vtk, PolyData
from sklearn.cluster import DBSCAN
import threading
import time
import trimesh
from typing import List

from mesh_data_converter import convert_pyvista_to_trimesh_mesh, convert_trimesh_mesh_to_pyvista, convert_pyvista_to_pymeshlab_mesh
from ring_neighbors_extractor import get_n_ring_neighbor_vertices_with_pyvista
from surface_characteristics import compute_surface_characteristics

def face_arrays(mesh: PolyData) -> List[np.ndarray]:
    cells = mesh.GetPolys()
    c = _vtk.vtk_to_numpy(cells.GetConnectivityArray())
    o = _vtk.vtk_to_numpy(cells.GetOffsetsArray())
    return np.split(c, o[1:-1])


def build_graph_from_mesh(pv_mesh):
    # Create a graph G with networkx using pyvista PolyData
    # Extract the mesh's vertices and faces
    vertices = pv_mesh.points
    faces = pv_mesh.faces.reshape(-1, 4)[:, 1:]

    # Create a list of edges from the faces
    edges = []
    for face in faces:
        for i in range(len(face)):
            edge = tuple(sorted([face[i], face[(i + 1) % len(face)]]))
            edges.append(edge)

    edges = list(set(edges))

    G = nx.Graph()

    # Add nodes and edges to the graph
    G.add_nodes_from(range(len(vertices)))
    G.add_edges_from(edges)

    return G


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


def counter_sort(List):
    counter_dict = Counter(List)

    # Only correct if our polygon is triangle
    if counter_dict.most_common(2)[0][0] == counter_dict.most_common(2)[0][1]:
        sorted_keys = sorted(counter_dict.keys(), key=lambda x: (counter_dict[x], x))
    else: 
        return counter_dict.most_common(1)[0][0]
    
    return sorted_keys[0]


def expand_boundary(pv_mesh, n, boundary_nodes):
    nth = get_n_ring_neighbor_vertices_with_pyvista(pv_mesh, n, True)

    new_boundary_nodes = []

    for vid in boundary_nodes:
        [new_boundary_nodes.append(v) for v in nth[vid]]

    new_boundary_nodes = np.unique(new_boundary_nodes)

    return new_boundary_nodes


def perform_hashing_vertices_list(pv_mesh, edges_largest):
    hash_points = [hash(p[0].item(), p[1].item(), p[2].item()) for p in pv_mesh.points]
    edge_hash_points = [hash(ep[0].item(), ep[1].item(), ep[2].item()) for ep in edges_largest.points]

    # Get boundary indices
    _, b_indices, _ = np.intersect1d(hash_points, edge_hash_points, return_indices=True)
   
    boundary_nodes = list(b_indices) 

    graph = build_graph_from_mesh(pv_mesh)

    possible_nodes = np.setdiff1d(list(graph.nodes()), boundary_nodes)
    
    return graph, boundary_nodes, possible_nodes


def collect_vertices_on_surface(graph, boundary_nodes, start_node):
    # Initial a value
    frag_face_id = 1

    frag_face_ids = set()
    surface_vertices = set()
    queue = [start_node]
    visited = set()

    while queue:
        node = queue.pop(0)

        visited.add(node)
        frag_face_ids.add(frag_face_id)
        surface_vertices.add(node)

        for neighbor in graph.neighbors(node):
            if neighbor not in visited and neighbor not in boundary_nodes:
                queue.append(neighbor)

        # Remove duplicate indices within queue
        queue = list(set(queue))

    return frag_face_ids, surface_vertices


def curvature_boundary(pv_mesh):
    curvature = pv_mesh.curvature(curv_type="Gaussian")
    threshold = 0.4
    high_curv_vertices = pv.PolyData(pv_mesh.points[(curvature > threshold).nonzero()[0]])

    return high_curv_vertices


def concat(a1, a2):
    if(np.size(a1) == 0):
        return a2
    elif(np.size(a2) == 0):
        return a1
    else:
        return np.concatenate((a1, a2)) 
    

# Progagate to get a fragment
def progagate(graph, startingId, faceFlag, faceID):
    # Set of all the vertice ids of the output fragment face
    cluster = set() 
    just_expanded  = [startingId]
    faceFlag[startingId] = faceID

    # Loop until for the previous step, no vertice added
    while np.size(just_expanded) > 0: 
        connecteds = []
        # Find all the adjacent vertices of the set just_expanded
        for vid in just_expanded: 
            #print("vid", vid)
            connected = [n for n in graph.neighbors(vid)]
            connecteds = concat(connecteds,connected)        
        connecteds = np.unique(connecteds, axis = None)
        #print("connecteds", connecteds)

        # Get only connecteds not in cluster
        just_expanded = [i for i in connecteds if i not in cluster]
        #print("just_expanded", just_expanded)
        for vid in just_expanded:
            cluster.add(vid)
            faceFlag[vid] = faceID
    return cluster


def save_propagated_surface(FILE_NAME, surface_vertices):
    surface_extracted = pv.PolyData(pv_mesh.points[list(surface_vertices)])
    surface_extracted.save(FILE_NAME)

    return surface_extracted


def find_sub_cluster(graph, start_node, point_labels, target_cluster):
    visited = set()
    stack = [start_node]
    cluster_edge = set()
    neighbor_color_edge = []
    connected_nodes = set()

    while stack:
        node = stack.pop()
        visited.add(node)

        if point_labels[node] == target_cluster:
            connected_nodes.add(node)

        neighbors = list(graph.neighbors(node))

        for neighbor in neighbors:
            if neighbor not in visited and point_labels[neighbor] == target_cluster:
                stack.append(neighbor)

        if node not in cluster_edge and any(point_labels[neighbors] != target_cluster):
            cluster_edge.add(node)

    neighbor_color_edge.extend(point_labels[y] for x in cluster_edge for y in graph.neighbors(x) if y not in connected_nodes)

    return connected_nodes, cluster_edge, neighbor_color_edge


def propagate_to_fill_hole(graph, point_labels, target_cluster, n_threshold):
    # Sub-cluster
    candidate_indices = [i for i in range(len(point_labels)) if point_labels[i] == target_cluster]
    
    possible_nodes = candidate_indices

    sub_cluster, sub_cluster_edge, ncolor = [], [], []

    while len(possible_nodes) > 0:
        start_node = random.choice(possible_nodes)
        c, e, nce = find_sub_cluster(graph, start_node, point_labels, target_cluster)
        c, e, nce = list(c), list(e), list(nce)
        sub_cluster.append(c)
        sub_cluster_edge.append(e)
        ncolor.append(nce)
        possible_nodes = np.setdiff1d(possible_nodes, c)

    # Compute and compare the length of each sub_cluster to a threshold
    lengths = np.array([len(sublist) for sublist in sub_cluster])
    short_sublist_indices = np.where(lengths < n_threshold)[0]

    # print(short_sublist_indices)

    most_common_plbl = [Counter(ncolor[i]).most_common(1)[0][0] if ncolor[i] else '0' for i in range(len(ncolor))]

    # 1 - Most common edge label approach
    for idx in short_sublist_indices:
        for nd in sub_cluster[idx]:
            point_labels[nd] = most_common_plbl[idx]

    # 2 - The n-ring neighbor label approach
    # for idx in short_sublist_indices:
    #     for nd in sub_cluster[idx]:
    #         # Check 7-ring neighbors
    #         n_cutoff = nx.single_source_shortest_path_length(graph, nd, cutoff=7)
    #         n_cutoff = list(n_cutoff.keys())
    #         n_cutoff.remove(nd)

    #         neighbor_stats = Counter(point_labels[n_cutoff])

    #         point_labels[nd] = max(neighbor_stats, key=neighbor_stats.get)

    return sub_cluster, sub_cluster_edge, point_labels


# %% Our implementation in which the bug exists

def custom_discrete_cmap(pv_mesh):
    # Define the number of steps
    num_steps = 20

    # Define the colors using linspace from red to blue
    colors = np.vstack((
        np.linspace(1.0, 0.0, num_steps),  # Red component
        np.zeros(num_steps),               # Green component
        np.linspace(0.0, 1.0, num_steps),  # Blue component
        np.ones(num_steps)                 # Alpha component
    )).T

    # Adding black color for outlier points
    colors = np.vstack((colors, np.array([0.0, 0.0, 0.0, 1.0])))

    # print(colors)

    mapping = np.linspace(pv_mesh["color_code"].min(), pv_mesh["color_code"].max(), 256)

    newcolors = np.empty((256, 4))

    for i in range(len(colors)):
        newcolors[mapping < (num_steps-2) - i] = colors[i]

    custom_cmap = ListedColormap(newcolors)

    return custom_cmap


def draw_pyvista_plotter(pv_mesh, normals, point_labels):
    # Render the edge lines on top of the original mesh.
    # Zoom in to provide a better figure.
    pv_mesh["color_code"] = point_labels

    p = pv.Plotter(shape=(1, 3))

    # Call custom colors
    my_cmap = custom_discrete_cmap(pv_mesh)

    p.subplot(0, 0)
    p.add_mesh(pv_mesh, scalars="cell", cmap='jet', copy_mesh=True) #, show_edges=True)
    #p.add_mesh(pv_mesh.glyph(geom=pv.Arrow(), orient='Normals'), color='black')
    p.add_axes(interactive=False)

    p.subplot(0, 1)
    p.add_points(normals, scalars=point_labels, cmap='jet')
    p.add_axes(interactive=False)

    p.subplot(0, 2)
    
    faces = pv_mesh.faces.reshape(-1, 4)[:, 1:]
    edges = []
    for face in faces:
        for i in range(len(face)):
            edge = tuple(sorted([face[i], face[(i + 1) % len(face)]]))
            edges.append(edge)

    lines = np.hstack([2*np.ones(len(edges), dtype=int).reshape(-1, 1), edges])

    p.add_mesh(pv.PolyData(normals, lines), scalars=point_labels, cmap='jet', style='wireframe')
    # p.add_mesh(pv.PolyData(normals, pv_mesh.faces), scalars=point_labels, cmap='jet') # Construct normal-domain sphere mesh
    # pv.PolyData(normals, pv_mesh.faces).save('sphere_sample.ply')
    p.add_axes(interactive=False)
    
    p.show()


# %% Main function

if __name__ == "__main__":

    file_name = './evaluation_dataset/Siggraph-06/venus_part07.obj'

    # Read mesh
    pv_mesh = pv.read(file_name)

    # Laplacian smooth
    # pv_mesh = pv_mesh.smooth(n_iter=20)

    graph = build_graph_from_mesh(pv_mesh)

    # [TEST POINT CLOUD NORMAL CLUSTERING] - DBSCAN 
    pv_mesh = fast_simplification.simplify_mesh(pv_mesh, target_reduction=0.9)
    
    pv_mesh = pv_mesh.smooth(n_iter=100)

    pv_mesh = pv_mesh.compute_normals(consistent_normals=False)

    normals = pv_mesh.point_normals

    ms = pymeshlab.MeshSet()
    ms.add_mesh(convert_pyvista_to_pymeshlab_mesh(pv_mesh))

    sharpness, roughness = compute_surface_characteristics(ms)

    data = np.hstack([normals, sharpness.reshape(-1, 1)]) # Combine normal with position information: ([normals, pv_mesh.points])

    # data = np.hstack([normals])
                  
    # Call DBSCAN
    eps = 0.175 # Adjust the epsilon (neighborhood distance) as needed
    # 0.125 with only normal
    min_samples = 10 # Adjust the minimum number of samples in a cluster as needed

    # dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    # clusters = dbscan.fit_predict(data)

    # labels = dbscan.labels_

    hierarchical_dbscan = hdbscan.HDBSCAN(min_cluster_size=min_samples)
    clusters = hierarchical_dbscan.fit_predict(data)

    labels = hierarchical_dbscan.labels_

    # Count how many clusters
    print(Counter(labels))
    total_group = Counter(labels).keys()

    # Access cluster labels for each point
    point_labels = clusters

    # Get the points (vertices) of the original mesh
    points = pv_mesh.points

    faces = pv_mesh.faces

    #print("Old label: ", point_labels)

    # for smallest_cluster in np.unique(point_labels):
    #     sub_cluster, sub_cluster_edge, point_labels = propagate_to_fill_hole(graph, point_labels, smallest_cluster, 10)

    #print("New label: ", point_labels)

    # Display in non-interpolated mode    
    cell_data = np.zeros(pv_mesh.n_cells)

    for c in range(pv_mesh.n_cells):
        l = list(np.array(point_labels)[pv_mesh.get_cell(c).point_ids])
        cell_data[c] = counter_sort(l).item()

    cell_data = cell_data.astype(int)

    pv_mesh["cell"] = cell_data 

    print(point_labels, cell_data)

    # Set frag_face_id for mesh clipping
    pv_mesh["frag_face_id"] = point_labels

    # Uncomment if need to save sub_mesh
    for requested_group in total_group:
        cid = np.where(np.array(cell_data) == requested_group)[0]

        # binary_vector = [1 if x == requested_group else 0 for x in cell_data]

        # pv_mesh["frag_face_id"] = binary_vector

        # Create a new PolyData mesh from the selected points (result as a point cloud)
        #sub_mesh = pv.PolyData(points[np.where(point_labels == requested_group)[0]])

        sub_mesh = pv.PolyData.from_regular_faces(points, faces.reshape(-1, 4)[:, 1:][cid])

        # Clip mesh using point label (result as a mesh)
        # sub_mesh = pv_mesh.clip_scalar(scalars="frag_face_id", invert=False, value=0.5)

        # Optionally, save the sub-mesh to a file
        # sub_mesh.save(file_name[:-4] + '-[' + str(requested_group) + ']-' + 'DBSCAN-' + str(eps) + '-' + str(min_samples) + '-' + str(time.time()) + '.ply')

    draw_pyvista_plotter(pv_mesh, normals, point_labels)