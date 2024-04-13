#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Author       : HUYNH Vinh-Nam, NGUYEN Hoang-Ha
# Email        : huynh-vinh.nam@usth.edu.vn
# Created Date : 09-November-2023
# Description  : 
"""
    Multi-spaces DBSCAN.
"""
#----------------------------------------------------------------------------
import argparse

from collections import defaultdict, Counter

import fast_simplification

# import hdbscan
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import multiprocessing
import networkx as nx
import numpy as np
import random
import os
# import pymeshlab
import pyvista as pv
from pyvista import examples
from pyvista import _vtk, PolyData
import scipy
from sklearn.cluster import DBSCAN
import time
import trimesh
from typing import List
import vtk

from generate_cmap import distinguishable_colors
from local_resolution import local_point_cloud_resolution
from mesh_data_converter import convert_pyvista_to_trimesh_mesh, convert_trimesh_mesh_to_pyvista, convert_pyvista_to_pymeshlab_mesh
from ring_neighbors_extractor import get_n_ring_neighbor_vertices_with_pyvista
from surface_characteristics import compute_surface_characteristics

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


# %% Callback implementation

def callback(point):
    global dict_coor_id
    global vert
    global cluster_labels
    p.add_point_labels(point, [f"{point}"])


# %% Misc functions
    
def counter_sort(List):
    counter_dict = Counter(List)

    # Only correct if our polygon is triangle
    if counter_dict.most_common(2)[0][0] == counter_dict.most_common(2)[0][1]:
        sorted_keys = sorted(counter_dict.keys(), key=lambda x: (counter_dict[x], x))
    else: 
        return counter_dict.most_common(1)[0][0]
    
    return sorted_keys[0]


def sigmoid(x):
    return 1 / (1 + np.exp(-x * 0.2))


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

    most_common_plbl = [Counter(ncolor[i]).most_common(1)[0][0] for i in range(len(ncolor))]
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

    mapping = np.linspace(pv_mesh["color_code"].min(), pv_mesh["color_code"].max(), 256)

    newcolors = np.empty((256, 4))

    for i in range(len(colors)):
        newcolors[mapping < (num_steps-2) - i] = colors[i]

    custom_cmap = ListedColormap(newcolors)

    return custom_cmap


def get_proper_neighbors(characteristic, radius):
    # NOTE:
    # Param: characteristic ~ vertices, normals, curvatures array
    #        radius ~ eps threshold
    characteristic_kdtree = scipy.spatial.cKDTree(characteristic)

    characteristic_neighbors_list = []

    for c in characteristic:
        neighbors_indices = characteristic_kdtree.query_ball_point(c, r=radius)
        characteristic_neighbors_list.append(neighbors_indices)

    return characteristic_neighbors_list


def multi_spaces_dbscan_cluster(pv_mesh, feature_type, eps2 = 0.16, eps3 = 3, eps4 = 2):

    # NOTE: Multi-spaces epsilons
    # Param: Vertices dist [eps1]
    #        Normals dist [eps2]
    #        Consider n-th ring as an epsilon constraint [eps3]
    #        Curvature dist [eps4]
    eps1 = 0.12

    # [-] Spatial Domain
    vert = pv_mesh.points

    # [-] Normal Domain
    normals = pv_mesh.point_normals

    # [-] Curvature information
    # Call pyvista mesh curvature computation
    k1 = pv_mesh.curvature(curv_type='maximum')
    k2 = pv_mesh.curvature(curv_type='minimum')

    # Combine two principle curvatures 
    # curv = np.abs(k1) + np.abs(k2)
    curv = np.maximum(k1, k2)

    minPts = 4

    '''
    # Vertices KDTree
    vert_neighbors_list = get_proper_neighbors(vert, eps1)

    print('[PRE-COMPUTE] All vertices distance')
    print('============')
    '''

    # Computing options
    if feature_type == 1:
        # We untouch the normals 
        modifier = normals

    if feature_type == 2:
        # curv = pv_mesh.curvature(curv_type='gaussian')

        # modifier = sigmoid(curv)
        modifier = np.tanh(curv * 0.05) - 0.5

        print('[PRE-COMPUTE] All curvature normalized')
        print('============')

        print("Before: \n", curv)
        print("After: \n", modifier)

        # Fuse curvature value with Normal vector (using scalar value of curvature to multiply with the radius 1 of the normal vector)
        # normals = np.multiply(normals.T, modifier).T
        normals = np.multiply(normals.T, np.add(1, modifier * 0.1)).T

    # Normal KDTree
    norm_neighbors_list = get_proper_neighbors(normals, eps2)

    print('[PRE-COMPUTE] All normals distance')
    print('============')

    nth = get_n_ring_neighbor_vertices_with_pyvista(pv_mesh, eps3, False)

    # print(len(norm_neighbors_list), len(nth)) # len(vert_neighbors_list)

    # If need to find common
    common = [list(filter(lambda x: x in nth[i], norm_neighbors_list[i])) for i in range(len(vert))]

    print("Before: ", common)

    if feature_type == 3:
        # Curv KDTree
        modifier = curv
        modifier = modifier.reshape(-1, 1)

        curv_neighbors_list = get_proper_neighbors(modifier, eps4)

        print('[PRE-COMPUTE] All curvature distance')
        print('============')

        # Intersection between the 2 sets: Normal KDTree and Curv KDTree
        common = [list(filter(lambda x: x in common[i], curv_neighbors_list[i])) for i in range(len(vert))]

    print("After: ", common)

    try:
        print('Resolution: ', np.mean(local_point_cloud_resolution(vert, 30)))
    except:
        print('Too small for computing local resolution...')

    # Initialization for the DBSCAN algorithm
    core_points = []
    border_points = []
    outlier_points = []
    for i in range(len(vert)):
        if len(common[i]) > minPts: # if len(common[i]) >= minPts:
            core_points.append(i)
        elif len(common[i]) > 0:
            border_points.append(i)
        else:
            outlier_points.append(i)

    print("outlier_points", outlier_points)

    # [MAIN ALGORITHM]
    # Initialize cluster labels
    cluster_labels = np.zeros(len(vert), dtype=int)
    unvisited = set([i for i in range(len(vert))])

    for vi in outlier_points:
        cluster_labels[vi] = -10 # outlier group
        unvisited.remove(vi)

    current_cluster_id = 1    
    queue = []
    while len(unvisited) > 0:
        p = unvisited.pop()
        
        if p in core_points:        
            cluster_labels[p] = current_cluster_id                                        
            neighbor_cores = common[p]
            for nb in neighbor_cores:                
                if(cluster_labels[nb] == 0 and nb not in outlier_points and nb in unvisited):                                                           
                    cluster_labels[nb] = current_cluster_id
                    unvisited.remove(nb)
                    
                    queue.append(nb)      
                
            while len(queue) != 0:
                pp = queue.pop(0)      
                neighbor_cores = common[pp]
                for nb in neighbor_cores:
                    if(cluster_labels[nb] == 0 and nb not in queue and nb in unvisited):                                                
                        cluster_labels[nb] = current_cluster_id
                        
                        unvisited.remove(nb)
                        queue.append(nb)
                        
            current_cluster_id +=1

    # print(cluster_labels)

    return curv, modifier, normals, cluster_labels


def compute_normal_representation(normals, point_normals):

    average_dict = {}
    
    clustered_dict = defaultdict(list)

    for idx, val in enumerate(point_normals):
        clustered_dict[val].append(normals[idx])

    for key, val in clustered_dict.items():
        if isinstance(val, list):
            average_dict[key] = np.mean(val, axis=0)
        else:
            average_dict[key] = val

    return average_dict 


def process_mesh(file_name, feature_type):
    
    print('[PROCESSING] Current file: ', file_name)
    print('============')

    # Read mesh
    pv_mesh = pv.read(file_name)

    pv_mesh = fast_simplification.simplify_mesh(pv_mesh, target_reduction=0.9)

    pv_mesh = pv_mesh.smooth(n_iter=100)

    graph = build_graph_from_mesh(pv_mesh)
    
    # [CALL CLUSTERING ALGORITHM] - DBSCAN
    curv, modifier, normals, point_labels = multi_spaces_dbscan_cluster(pv_mesh, feature_type)

    pv_mesh["color_code"] = point_labels

    # pv_mesh["scalar"] = pv_mesh.curvature(curv_type='minimum')

    my_cmap = custom_discrete_cmap(pv_mesh)

    for smallest_cluster in np.unique(point_labels):
        sub_cluster, sub_cluster_edge, point_labels = propagate_to_fill_hole(graph, point_labels, smallest_cluster, 10)
    
    # Uncomment if need to save sub_mesh
    total_group = Counter(point_labels).keys()

    for requested_group in total_group:
        binary_vector = [1 if x == requested_group else 0 for x in point_labels]

        pv_mesh["frag_face_id"] = binary_vector

        # Create a new PolyData mesh from the selected points (result as a point cloud)
        #sub_mesh = pv.PolyData(points[np.where(point_labels == requested_group)[0]])

        # Clip mesh using point label (result as a mesh)
        sub_mesh = pv_mesh.clip_scalar(scalars="frag_face_id", invert=False, value=0.5)

        # Optionally, save the sub-mesh to a file
        # sub_mesh.save(file_name[:-4] + '-[' + str(requested_group) + ']-' + 'DBSCAN_new_implementation-' + str(time.time()) + '.ply')

    return pv_mesh, curv, modifier, total_group, normals, point_labels


def draw_curvature_hist(file_name, curv, modifier):
    # Plot original and normalized distributions
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title('Original Half-Normal Distribution')
    plt.hist(curv, bins=30, color='blue', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.title('Normalized Distribution')
    plt.hist(modifier, bins=30, color='green', alpha=0.7)
    plt.xlabel('Normalized Value')
    plt.ylabel('Frequency')

    plt.tight_layout()

    plt.savefig(file_name[:-4] + '.png')

    # plt.show()


def draw_pyvista_plotter(pv_mesh, curv, normals, point_labels):
    # Render the edge lines on top of the original mesh.
    # Zoom in to provide a better figure.
    # pv_mesh["color_code"] = point_labels

    pv_mesh["curv_map"] = curv

    p = pv.Plotter(shape=(2, 2))

    # Call custom colors
    my_cmap = custom_discrete_cmap(pv_mesh)

    p.subplot(0, 0)
    p.add_mesh(pv_mesh, scalars="color_code", cmap='jet', copy_mesh=True) #, show_edges=True)
    # p.enable_surface_point_picking(callback=callback, show_point=False)
    p.add_axes(interactive=False)

    p.subplot(0, 1)
    cmin = np.percentile(curv, 8)
    cmax = np.percentile(curv, 95)
    p.add_mesh(pv_mesh, scalars="curv_map", clim=[cmin, cmax], cmap='jet', copy_mesh=True)

    p.add_axes(interactive=False)

    p.subplot(1, 0)
    p.add_points(normals, scalars=point_labels, cmap='jet')
    p.add_axes(interactive=False)

    p.subplot(1, 1)
    
    faces = pv_mesh.faces.reshape(-1, 4)[:, 1:]
    edges = []
    for face in faces:
        for i in range(len(face)):
            edge = tuple(sorted([face[i], face[(i + 1) % len(face)]]))
            edges.append(edge)

    lines = np.hstack([2*np.ones(len(edges), dtype=int).reshape(-1, 1), edges])

    p.add_mesh(pv.PolyData(normals, lines), scalars=point_labels, cmap='jet', style='wireframe')
    # p.add_mesh(pv.PolyData(normals, pv_mesh.faces), scalars=point_labels, cmap='jet') # Construct normal-domain sphere mesh
    pv.PolyData(normals, pv_mesh.faces).save('sphere_sample.ply')
    p.add_axes(interactive=False)
    
    # p.show()


def draw_animation(file_name, pv_mesh):
    p = pv.Plotter(off_screen=True)
    p.add_mesh(pv_mesh, scalars="color_code", cmap='jet', copy_mesh=True, lighting=False)
    
    path = p.generate_orbital_path(n_points=36, shift=pv_mesh.length)

    print(file_name[:-4] + ".gif")

    p.open_gif(file_name[:-4] + ".gif")
    p.orbit_on_path(path, write_frames=True)
    p.close()

# %% Main function

if __name__ == "__main__":

    vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
    
    parser = argparse.ArgumentParser(description='Process a 3D mesh file.')
    parser.add_argument('--file', type=str, help='Path to the OBJ file')
    parser.add_argument('--folder', type=str, help='Path to the folder containing OBJ files')
        
    curv_args_group = parser.add_mutually_exclusive_group()

    curv_args_group.add_argument('--norm_only', action='store_true', help='Segment using normal vectors only')
    curv_args_group.add_argument('--curv_fuse', action='store_true', help='Segment using normal vectors with fused curvature information')
    curv_args_group.add_argument('--curv_separate', action='store_true', help='Segment using normal vectors and curvature information separately')

    args = parser.parse_args()

    print(args)

    # By default, we cluster the normal domain (even the --norm_only flag is not declared)
    feature_type = 1
    
    if args.norm_only:
        feature_type = 1
    if args.curv_fuse:
        feature_type = 2
    if args.curv_separate:
        feature_type = 3

    if args.file:
        pv_mesh, curv, modifier, total_group, normals, point_labels = process_mesh(args.file, feature_type)
        
        cell_data = np.zeros(pv_mesh.n_cells)

        for c in range(pv_mesh.n_cells):
            l = list(np.array(point_labels)[pv_mesh.get_cell(c).point_ids])
            cell_data[c] = counter_sort(l).item()

        pv_mesh["color_code"] = cell_data 

        draw_animation(args.file, pv_mesh)
        p1 = multiprocessing.Process(target=draw_curvature_hist, args=(args.file, curv, modifier))
        p2 = multiprocessing.Process(target=draw_pyvista_plotter, args=(pv_mesh, curv, normals, point_labels))

        p1.start()
        p2.start()

        p1.join()
        p2.join()
    
    elif args.folder:
        obj_folder = args.folder

        for file_name in os.listdir(obj_folder):
            if file_name.endswith(".obj"):
                full_path = os.path.join(obj_folder, file_name)
                
                pv_mesh, curv, modifier, total_group, normals, point_labels = process_mesh(full_path, feature_type)
                
                cell_data = np.zeros(pv_mesh.n_cells)
                
                for c in range(pv_mesh.n_cells):
                    l = list(np.array(point_labels)[pv_mesh.get_cell(c).point_ids])
                    cell_data[c] = counter_sort(l).item()

                pv_mesh["color_code"] = cell_data 

                draw_animation(full_path, pv_mesh)
                p1 = multiprocessing.Process(target=draw_curvature_hist, args=(file_name, curv, modifier))
                p2 = multiprocessing.Process(target=draw_pyvista_plotter, args=(pv_mesh, curv, normals, point_labels))

                p1.start()
                p2.start()

                p1.join()
                p2.join()
                
    else:
        print("Please provide either --file or --folder flag.")