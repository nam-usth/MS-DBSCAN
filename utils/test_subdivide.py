import pyvista as pv

pv_mesh = pv.read('./evaluation_dataset/Chen-22/GT/icosahedron_subdivide/icosahedron_seg_20.obj')
pv_mesh = pv_mesh.triangulate()

pv_mesh = pv_mesh.subdivide(1)

# plotter = pv.Plotter()
# plotter.add_mesh(pv_mesh, show_edges=True)
# plotter.show()

pv.save_meshio('./evaluation_dataset/Chen-22/GT/icosahedron_subdivide_latest/icosahedron_latest_seg_20.obj', pv_mesh)