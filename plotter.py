import pyvista as pv

malla = pv.read("malla.vtk")

plotter = pv.Plotter(shape=(1, 2))

sel = 1

if (sel == 1): #Campo Electrico
    
    plotter.subplot(0, 0)
    plotter.add_mesh(malla, show_edges=False, cmap='jet', scalars='Campo_Electrico')

    plotter.subplot(0, 1)
    plotter.add_arrows(malla.points, malla['Campo_Electrico_Vectorial'], mag=0.004, cmap='jet')

if (sel == 2): #Campo Magnético
    
    plotter.subplot(0, 0)
    plotter.add_mesh(malla, show_edges=False, cmap='jet', scalars='Campo_Magnético')

    plotter.subplot(0, 1)
    plotter.add_arrows(malla.points, malla['Campo_Magnético_Vectorial'], mag=0.004, cmap='jet')
    
plotter.show_grid()
plotter.show()