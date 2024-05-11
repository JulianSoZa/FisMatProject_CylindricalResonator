import numpy as np
from scipy.special import jn, yn
import pyvista as pv
import meshio

f = 10.48E9
mu0 = 4*np.pi*1E-7
mur = 1
epsilon0 = 8.854E-12
epsilonr = 1
k = 2*np.pi*f*np.sqrt(mu0*mur*epsilon0*epsilonr)
eta = np.sqrt((mu0*mur)/(epsilon0*epsilonr))

a = 0.012
d = 0.02

Aplus = 1
H0 = -2*Aplus

dis = 50
rho = np.linspace(0.000000001, a, dis+1)
phi = np.linspace(0, 2*np.pi/4, dis+1)
z = np.linspace(0, d/2, dis+1)

beta = np.sqrt(k**2 - (1.841/a)**2)

puntos = []
celdas = []
Erho = []
Ephi = []
Hrho = []
Hphi = []
Hz = []

E = []
H = []

for i in enumerate(rho):
    for j in enumerate(phi):
        for g in enumerate(z):
            x = i[1] * np.cos(j[1])
            y = i[1] * np.sin(j[1])
            
            puntos.append([x, y, g[1]])
            Erho_m = ((k*eta*(a**2)*H0)/((1.841**2)*i[1])) * jn(1, 1.841*i[1]/a) * np.sin(j[1]) * np.sin(np.pi*g[1]/d)
            Ephi_m = ((k*eta*(a)*H0)/((1.841))) * ((jn(0, 1.841*i[1]/a)-jn(2, 1.841*i[1]/a))/2) * np.cos(j[1]) * np.sin(np.pi*g[1]/d)
            Erho.append(Erho_m)
            Ephi.append(Ephi_m)
            E.append(np.sqrt(Ephi_m**2 + Erho_m**2))
            
            Hrho_m = (beta*a*H0/1.841)*((jn(0, 1.841*i[1]/a)-jn(2, 1.841*i[1]/a))/2)* np.cos(j[1]) * np.cos(np.pi*g[1]/d)
            Hphi_m = ((-beta*(a**2)*H0)/((1.841**2)*i[1])) * jn(1, 1.841*i[1]/a) * np.sin(j[1]) * np.cos(np.pi*g[1]/d)
            Hz_m = H0 * jn(1, 1.841*i[1]/a) * np.cos(j[1]) * np.sin(np.pi*g[1]/d)
            Hrho.append(Hrho_m)
            Hphi.append(Hphi_m)
            Hz.append(Hz_m)
            
            H.append(np.sqrt(Hrho_m**2 + Hphi_m**2 + Hz_m**2))
            

for i in range(len(rho)-1):
    for j in range(len(phi)-1):
        for g in range(len(z)-1):
            
            n1 = np.ravel_multi_index((g, j, i), (dis+1, dis+1, dis+1))
            n2 = np.ravel_multi_index((g+1, j, i), (dis+1, dis+1, dis+1))
            n3 = np.ravel_multi_index((g+1, j+1, i), (dis+1, dis+1, dis+1))
            n4 = np.ravel_multi_index((g+0, j+1, i), (dis+1, dis+1, dis+1))
            n5 = np.ravel_multi_index((g, j, i+1), (dis+1, dis+1, dis+1))
            n6 = np.ravel_multi_index((g+1, j, i+1), (dis+1, dis+1, dis+1))
            n7 = np.ravel_multi_index((g+1, j+1, i+1), (dis+1, dis+1, dis+1))
            n8 = np.ravel_multi_index((g, j+1, i+1), (dis+1, dis+1, dis+1))
            celdas.append([n1, n2, n3, n4, n5, n6, n7, n8])

malla = meshio.Mesh(puntos, {"hexahedron": celdas})

malla.point_data['Campo_Electrico'] = E
malla.point_data['Campo_Magnético'] = H

original_mesh_pv = pv.wrap(malla)

original_mesh_pv.save("malla.vtk")

labels = dict(xtitle='X', ytitle='Y')

pl1 = pv.Plotter()
pl1.add_mesh(original_mesh_pv, show_edges=False, cmap='viridis', scalars='Campo_Electrico')
pl1.show_grid()
pl1.show()