import pyvista as pv
import numpy as np

# Crear una malla de ejemplo
x, y, z = np.mgrid[-1:1:20j, -1:1:20j, -1:1:20j]
vectors = np.empty((20, 20, 20, 3))
vectors[..., 0] = np.sin(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * z)  # Componente x del vector
vectors[..., 1] = np.sin(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * z)  # Componente y del vector
vectors[..., 2] = np.sin(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * z)  # Componente z del vector

# Crear la malla estructurada
grid = pv.StructuredGrid(x, y, z)

# Agregar los vectores a la malla como datos
grid['vectors'] = vectors.reshape(-1, 3)

# Calcular la magnitud de los vectores
magnitudes = np.linalg.norm(vectors, axis=-1)

# Crear el plotter y agregar la malla con flechas
plotter = pv.Plotter()
plotter.add_mesh(grid, scalars=magnitudes.ravel(), point_size=5.0)
plotter.add_arrows(grid.points, grid['vectors'], mag=0.2, color='white')

plotter.show()