import numpy as np
import pyvista as pv
import pandas as pd
from modules import conductivity, TEnml

# Modo TEnml
n = 1
m = 1
l = 1

#Dimensiones del cilindro (donde 'a' es el diametro y 'd' es la altura)
a = 0.012
d = 0.02

#Características dieléctricas del cilindro
mur = 1
epsilonr = 1

#Amplitud arbitraria de la onda viajera hacia adelante
Aplus = 1

#Material de la tapa al cual se le va a medir la conductividad
df = pd.read_json("data/measurements.json")

material = df.loc['Sheet, 304 stainless','Steel']

#material = {"f0": 10.47511340E9, "f1": 10.47236750E9, "f2": 10.47790310E9}

#Factor de calidad de descarga simulado
Q_un = 7800
#Factor de calidad del dispositivo con el mismo material para toda su estructura
Q_load0 = 2634.5


#Parametros de discretización de la malla
dis_rho = 32
dis_phi = 64
dis_z = 32

dis_rho_vec = 4
dis_phi_vec = 4
dis_z_vec = 2

#Generación de la malla
malla = TEnml.TEnml_modes(n, m, l, a, d, mur, epsilonr, Aplus, dis_rho, dis_phi, dis_z, dis_rho_vec, dis_phi_vec, dis_z_vec)

#malla.save("malla.vtk")

#Parametros de la electrotecnia
TSenergy, powerl, Q_meas, Q_0, Q, sigma = conductivity.material_conductivity(n, m, l, a, d, mur, epsilonr, Aplus, material, Q_un, Q_load0)


#Grafica de la malla
plotter = pv.Plotter(shape=(1, 2))

sel = 1

if (sel == 1): #Campo Electrico
    
    plotter.subplot(0, 0)
    plotter.add_mesh(malla, show_edges=False, cmap='jet', scalars='Campo_Electrico')

    plotter.subplot(0, 1)
    plotter.add_arrows(malla.points, malla['Campo_Electrico_Vectorial'], mag=0.0006, cmap='jet')
    #plotter.add_arrows(malla.points, malla['Campo_Electrico_Vectorial'], mag=0.0015, cmap='jet', show_scalar_bar=False)
    #plotter.view_xy()

if (sel == 2): #Campo Magnético
    
    plotter.subplot(0, 0)
    plotter.add_mesh(malla, show_edges=False, cmap='jet', scalars='Campo_Magnético')

    plotter.subplot(0, 1)
    plotter.add_arrows(malla.points, malla['Campo_Magnético_Vectorial'], mag=0.0009, cmap='jet')
    #plotter.add_arrows(malla.points, malla['Campo_Magnético_Vectorial'], mag=0.0009, cmap='jet', show_scalar_bar=False)
    #plotter.view_xy()
    
plotter.show_grid()
plotter.show()