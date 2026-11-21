#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Author       : HUYNH Vinh-Nam
# Email        : huynh-vinh.nam@usth.edu.vn 
# Created Date : 20-December-2023
# Updated Date : 
# Description  : 
"""
    -- Qt prototype [Remastered] --
    Graphical User Interface (GUI) for 3D navigation
"""
#----------------------------------------------------------------------------

from collections import Counter, defaultdict
import csv
import datetime
import fast_simplification
from itertools import product
import os
from os import walk

from PyQt5 import QtCore
from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QIntValidator, QRegExpValidator, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, \
    QFileDialog, QListWidget, QPushButton, QRadioButton, QLineEdit, QGroupBox, QGridLayout, QTextEdit, QCheckBox, QComboBox

import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from scipy.spatial import ConvexHull
import sys
import time

sys.path.append('./utils')

# from domino_chain import merge_dominoes
from multi_spaces_dbscan import multi_spaces_dbscan_cluster, build_graph_from_mesh, counter_sort
from ring_neighbors_extractor import get_n_ring_neighbor_vertices_with_pyvista
from surface_propagator import propagate_to_fill_hole

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        print(os.getcwd())
        self.setWindowIcon(QIcon('./icon.png'))
        self.setWindowTitle("3D Mesh Segmentation with Multi-space DBSCAN")
        self.setGeometry(100, 100, 1802, 1120)
        self.setMinimumSize(1802, 1120)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create a QTabWidget
        self.tab_widget = QTabWidget()
        central_layout = QGridLayout(central_widget)
        central_layout.addWidget(self.tab_widget, 0, 0)

        self.initialize_tab1()
        self.initialize_tab2()

        self.load_default()


    def initialize_tab1(self):
        # First Tab
        tab1 = QWidget()
        self.tab_widget.addTab(tab1, 'Basic')

        self.hbox_layout = QHBoxLayout()
        tab1.setLayout(self.hbox_layout)

        # Create a QGroupBox
        self.group_box_left = QGroupBox()
        self.group_box_left.setStyleSheet('QGroupBox { border: none; padding-top: 16px; margin-top: 11px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px; }')
        self.group_box_left.setFixedWidth(450)
        self.group_left_layout = QVBoxLayout(self.group_box_left)

        # Add buttons to the grid layout in Tab 1
        file_frame = QFrame()
        file_frame_layout = QVBoxLayout(file_frame)

        file_list_group = QGroupBox(' 3D files list ')
        file_list_group.setStyleSheet('QGroupBox { border: 1px solid red; }')
        file_frame_layout.addWidget(file_list_group)
        file_list_group_layout = QVBoxLayout(file_list_group)

        self.CWD = os.getcwd()
        self.fol_dir = self.CWD
        self.extensions = [".obj", ".off"]

        filenames = [fi for fi in os.listdir(self.CWD) if fi.endswith(tuple(self.extensions))]

        self.file_list = QListWidget()
        self.file_list.addItems(filenames)
        self.file_list.currentRowChanged.connect(self.load_checkpoint)
        file_list_group_layout.addWidget(self.file_list)

        file_list_button_group_layout = QHBoxLayout()
        file_list_group_layout.addLayout(file_list_button_group_layout)

        self.choose_file_button = QPushButton('Choose a Folder')
        self.choose_file_button.setMinimumWidth(150)
        self.choose_file_button.setMaximumWidth(200)
        self.choose_file_button.clicked.connect(self.select_folder)
        file_list_button_group_layout.addWidget(self.choose_file_button)

        self.view_file_button = QPushButton('Preview')
        self.view_file_button.setMinimumWidth(150)
        self.view_file_button.setMaximumWidth(200)
        self.view_file_button.clicked.connect(self.load_preview)
        file_list_button_group_layout.addWidget(self.view_file_button)

        #

        self.config_frame = QFrame()
        self.config_frame_layout = QGridLayout(self.config_frame)
        self.config_frame_layout.setSpacing(35)

        self.config_param_group = QGroupBox(' Settings ') # Original name: "Configuration"
        self.config_param_group.setStyleSheet('QGroupBox { border: 1px solid blue; }')

        self.parameter_panel = QGridLayout(self.config_param_group)

        self.sel_var = 1
        self.interpolate = False
        self.hole_filling = False
        self.show_edge = False
        self.command = 'preview'

        self.label_feature = QLabel('Feature type')
        self.label_interpolate = QLabel('Interpolate')
        self.label_hole_filling = QLabel('Hole filling')
        self.label_show_edge = QLabel('Show edge')

        self.option_method = QComboBox()

        self.option_method.addItems(["Normal only", "Curvature fuse", "Curvature separate"])
        self.option_method.currentIndexChanged.connect(self.feature_mode_selection)

        self.option_interpolate = QComboBox()
        self.option_interpolate.addItems(["Disable", "Enable"])
        self.option_interpolate.currentIndexChanged.connect(self.interpolate_mode_selection)

        self.option_hole_filling = QComboBox()
        self.option_hole_filling.addItems(["Disable", "Enable"])
        self.option_hole_filling.currentIndexChanged.connect(self.hole_filling_mode_selection)

        self.option_show_edge = QComboBox()
        self.option_show_edge.addItems(["Disable", "Enable"])
        self.option_show_edge.currentIndexChanged.connect(self.show_edge_mode_selection)

        self.parameter_panel.addWidget(self.label_feature, 0, 0)
        self.parameter_panel.addWidget(self.option_method, 0, 1)
        self.parameter_panel.addWidget(self.label_interpolate, 1, 0)
        self.parameter_panel.addWidget(self.option_interpolate, 1, 1)
        self.parameter_panel.addWidget(self.label_hole_filling, 2, 0)
        self.parameter_panel.addWidget(self.option_hole_filling, 2, 1)
        self.parameter_panel.addWidget(self.label_show_edge, 3, 0)
        self.parameter_panel.addWidget(self.option_show_edge, 3, 1)

        # self.config_frame_layout.addWidget(self.config_param_group)

        #

        self.threshold_label_frame = QGroupBox(' Thresholds ')
        self.threshold_label_frame.setStyleSheet('QGroupBox { border: 1px solid blue; }')
        # self.config_frame_layout.addWidget(self.threshold_label_frame)

        threshold_panel = QGridLayout(self.threshold_label_frame)
        self.label1 = QLabel('Eps normal')
        self.label2 = QLabel('Eps n-ring')
        self.label3 = QLabel('Eps curvature')
        
        float_validator = QRegExpValidator(QRegExp(r'[0-9]*\.?[0-9]+'))
        int_validator = QIntValidator()

        self.text1 = QLineEdit(self)
        self.text1.setMaximumWidth(300)
        self.text1.setValidator(float_validator)
        self.text2 = QLineEdit(self)
        self.text2.setMaximumWidth(300)
        self.text2.setValidator(int_validator)
        self.text3 = QLineEdit(self)
        self.text3.setMaximumWidth(300)
        self.text3.setValidator(float_validator)

        self.eps2 = 0.16
        self.eps3 = 3
        self.eps4 = 2

        threshold_button_group_layout = QHBoxLayout()

        self.button1 = QPushButton('Load default')
        self.button1.clicked.connect(self.load_default)
        threshold_button_group_layout.addWidget(self.button1)

        threshold_panel.addWidget(self.label1, 0, 0)
        threshold_panel.addWidget(self.text1, 0, 1)
        threshold_panel.addWidget(self.label2, 1, 0)
        threshold_panel.addWidget(self.text2, 1, 1)
        threshold_panel.addWidget(self.label3, 2, 0)
        threshold_panel.addWidget(self.text3, 2, 1)
        threshold_panel.addLayout(threshold_button_group_layout, 3, 0, 1, 2)

        #
        
        self.seg_label_frame = QGroupBox(' Surface segmentation ')
        self.seg_label_frame.setStyleSheet('QGroupBox { border: 1px solid red; }')
        self.config_frame_layout.addWidget(self.seg_label_frame)

        temp_panel = QGridLayout(self.seg_label_frame)

        temp_button_group_layout = QVBoxLayout()

        temp_button_group_layout.addWidget(self.config_param_group)
        temp_button_group_layout.addWidget(self.threshold_label_frame)

        self.button2 = QPushButton('Compute') # Original name: "Re-compute"
        self.button2.clicked.connect(self.compute)
        self.button2.setToolTip("Run segmentation")
        temp_button_group_layout.addWidget(self.button2)

        temp_panel.addLayout(temp_button_group_layout, 0, 0)

        #

        self.save_parameter_frame = QGroupBox(' Export parameter ')
        self.save_parameter_frame.setStyleSheet('QGroupBox { border: 1px solid red; }')
        self.config_frame_layout.addWidget(self.save_parameter_frame)

        save_parameter_layout = QHBoxLayout(self.save_parameter_frame)

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_to_file)
        self.save_button.setToolTip("Save all configurations and parameters to `data.csv`")
        save_parameter_layout.addWidget(self.save_button)

        #

        display_frame = QFrame()
        display_frame_layout = QVBoxLayout(display_frame)

        global plotter
        plotter = BackgroundPlotter(show=False, shape=(2,2))

        display_frame_layout.addWidget(plotter)

        self.group_left_layout.addWidget(file_frame)
        self.group_left_layout.addWidget(self.config_frame)

        # Add the group box to the central layout
        self.hbox_layout.addWidget(self.group_box_left)
        self.hbox_layout.addWidget(display_frame)


    def initialize_tab2(self):
        # Second Tab: Placeholder
        tab2 = QWidget()
        self.tab_widget.addTab(tab2, 'Help')


    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.fol_dir = folder
            self.update_listbox()


    def select_folder_2(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.fol_dir = folder
            self.update_listbox()


    def update_listbox(self):
        filenames = [fi for fi in os.listdir(self.fol_dir) if fi.endswith(tuple(self.extensions))]
        self.file_list.clear()
        self.file_list.addItems(filenames)


    def feature_mode_selection(self, value):
        self.sel_var = value + 1

        print(self.sel_var)
        
        # Reload plotter
        # self.compute_selection()


    def interpolate_mode_selection(self, value):
        self.interpolate = bool(value)
        
        # Reload plotter
        # self.compute_selection()


    def hole_filling_mode_selection(self, value):
        self.hole_filling = bool(value)

        # Reload plotter
        # self.compute_selection()


    def show_edge_mode_selection(self, value):
        self.show_edge = bool(value)

        # Reload plotter
        # self.compute_selection()


    def load_preview(self):
        value = self.file_list.currentItem().text()

        pv_mesh = pv.read(os.path.join(self.fol_dir, value))

        if "subdivide" in value:
            pv_mesh = pv_mesh
        else:
            pv_mesh = fast_simplification.simplify_mesh(pv_mesh, target_reduction=0.9)
        
            pv_mesh = pv_mesh.smooth(n_iter=100)

        self.previewer(pv_mesh)


    def compute_selection(self):
        value = self.file_list.currentItem().text()

        pv_mesh = pv.read(os.path.join(self.fol_dir, value))

        if "subdivide" in value:
            pv_mesh = pv_mesh
        else:
            pv_mesh = fast_simplification.simplify_mesh(pv_mesh, target_reduction=0.9)
        
            pv_mesh = pv_mesh.smooth(n_iter=100)

        # TODO: CACHING TECHNIQUE
        # Do not need to recompute again if the computation has already been made
        # Using dictionary to cache, key = eps2-eps3-eps4-feature_type-folder_name-file_name, value = pv_mesh object

        # Callback function 1 - Compute DBSCAN with multi-constraint
        pv_mesh, normals, edge_lines = self.compute_dbscan(pv_mesh)

        # Callback function 2 - Display results
        self.visualizer(pv_mesh, normals, edge_lines, self.interpolate)

    
    def load_default(self):
        self.text1.setText(str(0.16))
        self.text2.setText(str(3))
        self.text3.setText(str(2))


    def read_clustered_result(self, sckpt_filename):
        segments_dict = {}

        # Read the clustered result from the .sckpt file
        with open(sckpt_filename, 'r') as file:
            lines = file.readlines()
            segment_id = None
            for line in lines:
                line = line.strip()
                if line.startswith('MODE'):
                    self.option_method.setCurrentIndex(int(lines[3]) - 1)

                if line.startswith('THRESHOLDS'):
                    eps = [e.strip() for e in lines[5].strip('\n').strip('[]').split(',')]
                    self.text1.setText(eps[0])
                    self.text2.setText(eps[1])
                    self.text3.setText(eps[2])

                if line.startswith('SEGMENT'):
                    segment_id = line.split()[1]
                elif segment_id is not None:
                    vertex_ids = [int(vertex_id) for vertex_id in line.strip('[]').split(',')]
                    segments_dict.update({vertex_id: segment_id for vertex_id in vertex_ids})
                    segment_id = None
                   
        max_vertex_id = max(segments_dict.keys())
        segments_array = [segments_dict.get(vertex_id, -1) for vertex_id in range(max_vertex_id + 1)]
        return segments_array


    def load_dbscan(self, pv_mesh, segments_array):  
        graph = build_graph_from_mesh(pv_mesh)

        # [-] Rebuild normals domain
        normals = pv_mesh.point_normals

        # [-] Rebuild curvature information
        k1 = pv_mesh.curvature(curv_type='maximum')
        k2 = pv_mesh.curvature(curv_type='minimum')

        # Combine two principle curvatures 
        # curv = np.abs(k1) + np.abs(k2)
        curv = np.maximum(k1, k2)
        
        # [LOAD PRE-COMPUTED DATA] - DBSCAN
        point_labels = np.array(segments_array)

        print("[RESTORED] Loaded segmentation checkpoint\n", point_labels)

        if (self.hole_filling):
            for smallest_cluster in np.unique(point_labels):
                sub_cluster, sub_cluster_edge, point_labels = propagate_to_fill_hole(graph, point_labels, smallest_cluster, max(min(20, int(0.025 * len(pv_mesh.points))), 4))

        pv_mesh["color_code"] = point_labels

        cell_data = np.zeros(pv_mesh.n_cells)

        for c in range(pv_mesh.n_cells):
            l = list(np.array(point_labels)[pv_mesh.get_cell(c).point_ids])
            cell_data[c] = counter_sort(l).item()

        cell_data = cell_data.astype(int)

        pv_mesh["cell"] = cell_data 

        pv_mesh["curv_map"] = curv

        edge_lines = self.highlight_edge(pv_mesh)

        # Save mesh
        # self.save_mesh(pv_mesh, point_labels)

        return pv_mesh, normals, edge_lines

            
    def load_checkpoint(self):
        try:
            filename = self.file_list.currentItem().text()
            sckpt_filename = filename.split('.')[0] + '.sckpt'

            pv_mesh = pv.read(os.path.join(self.fol_dir, filename))

            if "subdivide" in filename:
                pv_mesh = pv_mesh
            else:
                pv_mesh = fast_simplification.simplify_mesh(pv_mesh, target_reduction=0.9)
            
                pv_mesh = pv_mesh.smooth(n_iter=100)

            if os.path.exists(os.path.join(self.fol_dir, sckpt_filename)):
                # Read clustered result from the .sckpt file
                segments_array = self.read_clustered_result(os.path.join(self.fol_dir, sckpt_filename))
                
                pv_mesh, normals, edge_lines = self.load_dbscan(pv_mesh, segments_array)

                self.visualizer(pv_mesh, normals, edge_lines, self.interpolate)
            else:
                self.previewer(pv_mesh)
        except:
            pass


    def save_checkpoint(self, point_labels):
        index_dict = defaultdict(list)

        for idx, val in enumerate(point_labels):
            index_dict[val].append(idx)

        segmentation_results = {
            'filename': self.file_list.currentItem().text(),
            'compute_mode': str(self.sel_var),
            'eps2': self.text1.text(),
            'eps3': self.text2.text(),
            'eps4': self.text3.text(),
            'clusters': index_dict
        }

        filename = self.file_list.currentItem().text()
        sckpt_filename = filename.split('.')[0] + '.sckpt'

        with open(os.path.join(self.fol_dir, sckpt_filename), 'w') as file:
            file.write(f"FILENAME\n{segmentation_results['filename']}\n")
            file.write(f"MODE\n{segmentation_results['compute_mode']}\n")
            file.write(f"THRESHOLDS\n[{segmentation_results['eps2']}, {segmentation_results['eps3']}, {segmentation_results['eps4']}]\n")

            for cluster, vertices in segmentation_results['clusters'].items():
                file.write(f"SEGMENT {cluster}\n")
                file.write(f"{vertices}\n")


    def save_to_file(self):
        text = self.fol_dir + ',' + self.file_list.currentItem().text() + ',' + str(self.sel_var) + ',' + self.text1.text() + '_' + self.text2.text() + '_' + self.text3.text()
        if text:
            try:
                with open('data.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([text])
            except Exception as e:
                print("Error:", e)


    def save_mesh(self, pv_mesh, point_labels):
        saved_path = os.path.join(self.fol_dir, 'segmented_results')

        if not os.path.exists(saved_path):
            os.makedirs(saved_path)

        total_group = np.unique(point_labels)

        # Set frag_face_id for mesh clipping in [interpolated mode]
        # pv_mesh["frag_face_id"] = point_labels

        # Set cell_data for mesh clipping in [non-interpolated mode]
        cell_data = np.zeros(pv_mesh.n_cells)

        for c in range(pv_mesh.n_cells):
            l = list(np.array(point_labels)[pv_mesh.get_cell(c).point_ids])
            cell_data[c] = counter_sort(l).item()

        cell_data = cell_data.astype(int)

        pv_mesh["cell"] = cell_data 

        for requested_group in total_group:
            # [NOTE] Uncomment for save in [interpolated mode]
            # binary_vector = [1 if x == requested_group else 0 for x in point_labels]

            # pv_mesh["frag_face_id"] = binary_vector

            # # Create a new PolyData mesh from the selected points (result as a point cloud)
            # #sub_mesh = pv.PolyData(points[np.where(point_labels == requested_group)[0]])

            # # Clip mesh using point label (result as a mesh)
            # sub_mesh = pv_mesh.clip_scalar(scalars="frag_face_id", invert=False, value=0.5)

            # [UPDATED] 18-Feb-2024
            # [NOTE] Save sub_mesh with cell values in [non-interpolated mode]
            cid = np.where(np.array(cell_data) == int(requested_group))[0]

            sub_mesh = pv.PolyData.from_regular_faces(pv_mesh.points, pv_mesh.faces.reshape(-1, 4)[:, 1:][cid])

            # Optionally, save the sub-mesh to a file
            sub_mesh.save(os.path.join(saved_path, self.file_list.currentItem().text().split('.')[0] + '-[' + str(requested_group) + ']-' + 'MS_DBSCAN-' + str(self.text1.text()) + '-' + str(self.text2.text()) + '-' + str(self.text3.text()) + '-' + str(self.sel_var) + '-' + str(time.time()) + '.ply'))


    def compute(self):
        self.eps2 = float(self.text1.text())
        self.eps3 = int(self.text2.text())
        self.eps4 = float(self.text3.text())

        self.compute_selection()


    def compute_dbscan(self, pv_mesh):  
        graph = build_graph_from_mesh(pv_mesh)
        
        # [CALL CLUSTERING ALGORITHM] - DBSCAN
        curv, modifier, normals, point_labels = multi_spaces_dbscan_cluster(pv_mesh, self.sel_var, self.eps2, self.eps3, self.eps4)

        # [CALL CHECKPOINT SAVING]
        self.save_checkpoint(point_labels)

        if (self.hole_filling):
            for smallest_cluster in np.unique(point_labels):
                sub_cluster, sub_cluster_edge, point_labels = propagate_to_fill_hole(graph, point_labels, smallest_cluster, max(min(20, int(0.025 * len(pv_mesh.points))), 4))

        pv_mesh["color_code"] = point_labels

        cell_data = np.zeros(pv_mesh.n_cells)

        for c in range(pv_mesh.n_cells):
            l = list(np.array(point_labels)[pv_mesh.get_cell(c).point_ids])
            cell_data[c] = counter_sort(l).item()

        cell_data = cell_data.astype(int)

        pv_mesh["cell"] = cell_data 

        pv_mesh["curv_map"] = curv

        edge_lines = self.highlight_edge(pv_mesh)

        # Save mesh
        # self.save_mesh(pv_mesh, point_labels)

        return pv_mesh, normals, edge_lines
 

    def highlight_edge(self, pv_mesh):
        edge_vertice, edge_lines = [], []

        if self.show_edge:
            edge_vertice = [
                idx for idx in range(len(pv_mesh.points))
                if len(set(pv_mesh["cell"][pv_mesh.extract_points(idx, adjacent_cells=True)["vtkOriginalCellIds"]])) != 1
            ]

            ring = get_n_ring_neighbor_vertices_with_pyvista(pv_mesh, 1, False)

            # Create the edge combination list
            combination = []

            for e in edge_vertice:
                result = [r for r in edge_vertice if set(ring[e]).__contains__(r)]
                combination.extend([[e, r] for r in result])

            # Remove mirror subsets while preserving order
            seen = set()
            combination_unique = []

            for subset in combination:
                tuple_subset = tuple(subset)
                
                # Check if the subset or its mirror is already in the set
                if tuple_subset not in seen and tuple(reversed(tuple_subset)) not in seen:
                    seen.add(tuple_subset)
                    combination_unique.append(list(tuple_subset))

            for count, c in enumerate(combination_unique):
                common_cell_ids = set(pv_mesh.point_cell_ids(c[0])).intersection(pv_mesh.point_cell_ids(c[1]))
                if len(set(pv_mesh["cell"][list(common_cell_ids)])) != 1:
                    edge_lines.append([c[0], c[1]])

                    print(f"[EDGE] Checking lines... {(count+1)}/{len(combination_unique)}")

            unique_edge_pairs = {tuple(sorted(pair)) for pair in edge_lines}
            unique_edge_pairs_list = [list(pair) for pair in unique_edge_pairs]

            # TODO: Join multiple straight lines into polylines (using domino algorithm)
            # NOTE: Currently unnecessary feature
            
            # result = []

            # for c in merge_dominoes(unique_edge_pairs_list):
            #     result.append([len(c)] + c)

            # print("[EDGE] Set of polylines: ", result)

            edge_lines = np.hstack((np.full((len(unique_edge_pairs_list), 1), 2), unique_edge_pairs_list))

        return edge_lines


    def previewer(self, pv_mesh):
        
        plotter.clear()
        plotter.enable_lightkit()

        # plotter.clear_actors()

        # plotter.renderer.clear_actors()
        
        plotter.subplot(0, 0)
        plotter.add_mesh(pv_mesh, name="mesh_preview", copy_mesh=True)
        

    def visualizer(self, pv_mesh, normals, edge_lines, interpolate=False):

        plotter.clear()
        plotter.enable_lightkit()

        # plotter.clear_actors()

        # plotter.renderer.clear_actors()

        plotter.subplot(0, 0)

        plotter.add_mesh(pv_mesh, color="lightblue", name="mesh_preview", copy_mesh=True)
        # plotter.add_axes(interactive=False)

        plotter.subplot(1, 1)

        # [PYVISTA PLOTTER] Camera setup for screenshot only
        # plotter.camera_position = [
        #     (0.01, 0.02, 0.03),
        #     (0.1, 0, -0.1),
        #     (-0.01, 0.02, -0.02),
        # ]

        # plotter.camera_position = [
        #     (-0.3914, -0.4542, -0.7670),
        #     (-0.0243, -0.0336, 0.0222),
        #     (0.2148, -0.8998, 0.3796),
        # ]

        if (interpolate==True):
            plotter.add_mesh(pv_mesh, scalars="color_code", cmap='jet', name="mesh_clustered", copy_mesh=True)
        else:
            plotter.add_mesh(pv_mesh, scalars="cell", cmap='jet', name="mesh_clustered", copy_mesh=True)

        if (self.show_edge==True):
            lines_poly = pv.PolyData(pv_mesh.points)
            lines_poly.lines = edge_lines

            plotter.add_mesh(lines_poly, color="pink", line_width=5, point_size=0.001)
        # plotter.add_axes(interactive=False)

        plotter.subplot(0, 1)
        cmin = np.percentile(pv_mesh["curv_map"], 8)
        cmax = np.percentile(pv_mesh["curv_map"], 95)
        plotter.add_mesh(pv_mesh, scalars="curv_map", clim=[cmin, cmax], cmap='jet', name="mesh_curv_map", copy_mesh=True)
        # plotter.add_axes(interactive=False)

        plotter.subplot(1, 0)
        plotter.add_mesh(normals, scalars=pv_mesh["color_code"], name="normal_pcd", cmap='jet', copy_mesh=True)
        # plotter.add_axes(interactive=False)
        # plotter.add_mesh(normals, name="normal_pcd", copy_mesh=True)

        # plotter.subplot(0, 0)

        # faces = pv_mesh.faces.reshape(-1, 4)[:, 1:]
        # edges = []
        # for face in faces:
        #     for i in range(len(face)):
        #         edge = tuple(sorted([face[i], face[(i + 1) % len(face)]]))
        #         edges.append(edge)

        # lines = np.hstack([2*np.ones(len(edges), dtype=int).reshape(-1, 1), edges])

        # plotter.add_mesh(pv.PolyData(normals, lines), scalars=pv_mesh["color_code"], name="normal_mesh", cmap='jet', style='wireframe')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())