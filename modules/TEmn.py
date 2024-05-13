import numpy as np
from scipy.special import jn, yn, jnp_zeros
import pyvista as pv
import meshio

m = 2
n = 2
l = 2

a = 0.012
d = 0.02

c = 299792458

mu0 = 4*np.pi*1E-7
mur = 1
epsilon0 = 8.854E-12
epsilonr = 1

ceros_jnp = jnp_zeros(n, 5)

pnm_p = ceros_jnp[m-1]

f = (c/(2*np.pi*np.sqrt(mur*epsilonr)))*np.sqrt((pnm_p/a)**2 + (l*np.pi/d)**2)

k = 2*np.pi*f*np.sqrt(mu0*mur*epsilon0*epsilonr)
eta = np.sqrt((mu0*mur)/(epsilon0*epsilonr))

Aplus = 1
H0 = -2*Aplus

beta = np.sqrt(k**2 - (pnm_p/a)**2)

dis = 20
rho = np.linspace(0.000000001, a, dis+1)
phi = np.linspace(0, 3*np.pi/2, dis+1)
z = np.linspace(0, d, dis+1)

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
            Erho_m = ((k*eta*(a**2)*H0)/((pnm_p**2)*i[1])) * jn(1, pnm_p*i[1]/a) * np.sin(n*j[1]) * np.sin(l*np.pi*g[1]/d)
            Ephi_m = ((k*eta*(a)*H0)/((pnm_p))) * ((jn(0, pnm_p*i[1]/a)-jn(2, pnm_p*i[1]/a))/2) * np.cos(n*j[1]) * np.sin(l*np.pi*g[1]/d)
            Erho.append(Erho_m)
            Ephi.append(Ephi_m)
            E.append(np.sqrt(Ephi_m**2 + Erho_m**2))
            
            Hrho_m = (beta*a*H0/pnm_p)*((jn(0, pnm_p*i[1]/a)-jn(2, pnm_p*i[1]/a))/2)* np.cos(n*j[1]) * np.cos(l*np.pi*g[1]/d)
            Hphi_m = ((-beta*n*(a**2)*H0)/((pnm_p**2)*i[1])) * jn(1, pnm_p*i[1]/a) * np.sin(n*j[1]) * np.cos(l*np.pi*g[1]/d)
            Hz_m = H0 * jn(1, pnm_p*i[1]/a) * np.cos(n*j[1]) * np.sin(l*np.pi*g[1]/d)
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

#original_mesh_pv.save("malla.vtk")

labels = dict(xtitle='X', ytitle='Y')

pl1 = pv.Plotter()
pl1.add_mesh(original_mesh_pv, show_edges=False, cmap='viridis', scalars='Campo_Magnético')
pl1.show_grid()
pl1.show()