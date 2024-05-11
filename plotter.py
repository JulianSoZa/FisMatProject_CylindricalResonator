import pyvista as pv

mesh = pv.read("malla.vtk")
pl1 = pv.Plotter()
pl1.add_mesh(mesh, show_edges=False, cmap='viridis', scalars='Campo_Electrico')
pl1.show()