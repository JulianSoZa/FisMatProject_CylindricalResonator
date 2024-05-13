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

dis_rho = 4
dis_phi = 15
dis_z = 9

rho = np.linspace(0.000000001, a, dis_rho+1)
phi = np.linspace(0, 3*np.pi/2, dis_phi+1)
z = np.linspace(0, d, dis_z+1)

puntos = []
celdas = []
Erho = []
Ephi = []
Hrho = []
Hphi = []
Hz = []

E = []
H = []

Evec = []
Hvec = []

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
            
            Evec.append([np.sqrt(Ephi_m**2 + Erho_m**2)*np.cos(j[1]), np.sqrt(Ephi_m**2 + Erho_m**2)*np.sin(j[1]), 0])
            Hvec.append([np.sqrt(Hrho_m**2 + Hphi_m**2 + Hz_m**2)*np.cos(j[1]), np.sqrt(Hrho_m**2 + Hphi_m**2 + Hz_m**2)*np.sin(j[1]), Hz_m])

for i in range(len(rho)-1):
    for j in range(len(phi)-1):
        for g in range(len(z)-1):
            
            n1 = np.ravel_multi_index((g, j, i), (dis_z+1, dis_phi+1, dis_rho+1))
            n2 = np.ravel_multi_index((g+1, j, i), (dis_z+1, dis_phi+1, dis_rho+1))
            n3 = np.ravel_multi_index((g+1, j+1, i), (dis_z+1, dis_phi+1, dis_rho+1))
            n4 = np.ravel_multi_index((g+0, j+1, i), (dis_z+1, dis_phi+1, dis_rho+1))
            n5 = np.ravel_multi_index((g, j, i+1), (dis_z+1, dis_phi+1, dis_rho+1))
            n6 = np.ravel_multi_index((g+1, j, i+1), (dis_z+1, dis_phi+1, dis_rho+1))
            n7 = np.ravel_multi_index((g+1, j+1, i+1), (dis_z+1, dis_phi+1, dis_rho+1))
            n8 = np.ravel_multi_index((g, j+1, i+1), (dis_z+1, dis_phi+1, dis_rho+1))
            celdas.append([n1, n2, n3, n4, n5, n6, n7, n8])

malla = meshio.Mesh(puntos, {"hexahedron": celdas})

original_mesh_pv = pv.wrap(malla)

original_mesh_pv['Campo_Electrico_Vectorial'] = np.array(Evec)/np.max(Evec)
original_mesh_pv['Campo_Magnético_Vectorial'] = np.array(Hvec)/np.max(Hvec)

mesh_s = pv.read("malla.vtk")

labels = dict(xtitle='X', ytitle='Y')

plotter = pv.Plotter(shape=(1, 2))

plotter.subplot(0, 0)
#plotter.add_mesh(mesh_s, show_edges=False, cmap='viridis', scalars='Campo_Magnético')
plotter.add_mesh(mesh_s, show_edges=False, cmap='viridis', scalars='Campo_Electrico')

plotter.subplot(0, 1)
#plotter.add_arrows(original_mesh_pv.points, original_mesh_pv['Campo_Magnético_Vectorial'], mag=0.003, cmap='viridis')
plotter.add_arrows(original_mesh_pv.points, original_mesh_pv['Campo_Electrico_Vectorial'], mag=0.003, cmap='viridis')

plotter.show_grid()
plotter.show()