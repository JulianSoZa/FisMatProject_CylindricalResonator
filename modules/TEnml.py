import numpy as np
from scipy.special import jn, yn, jnp_zeros
import pyvista as pv
import meshio

def TEnml_modes(n, m, l, a, d, mur, epsilonr, Aplus, dis_rho, dis_phi, dis_z, dis_rho_vec, dis_phi_vec, dis_z_vec):
    
    c = 299792458
    
    mu0 = 4*np.pi*1E-7
    epsilon0 = 8.854E-12
    
    ceros_jnp = jnp_zeros(n, 10)
    pnm_p = ceros_jnp[m-1]
    
    f = (c/(2*np.pi*np.sqrt(mur*epsilonr)))*np.sqrt((pnm_p/a)**2 + (l*np.pi/d)**2)

    k = 2*np.pi*f*np.sqrt(mu0*mur*epsilon0*epsilonr)
    eta = np.sqrt((mu0*mur)/(epsilon0*epsilonr))
    
    H0 = -2*Aplus
    
    beta = np.sqrt(k**2 - (pnm_p/a)**2)

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
                
                if((i[0]<(len(rho)-1))&(j[0]<(len(phi)-1))&(g[0]<(len(z)-1))):
                    n1 = np.ravel_multi_index((i[0], j[0], g[0]), (dis_rho+1, dis_phi+1, dis_z+1))
                    n2 = np.ravel_multi_index((i[0]+1, j[0], g[0]), (dis_rho+1, dis_phi+1, dis_z+1))
                    n3 = np.ravel_multi_index((i[0]+1, j[0]+1, g[0]), (dis_rho+1, dis_phi+1, dis_z+1))
                    n4 = np.ravel_multi_index((i[0], j[0]+1, g[0]), (dis_rho+1, dis_phi+1, dis_z+1))
                    n5 = np.ravel_multi_index((i[0], j[0], g[0]+1), (dis_rho+1, dis_phi+1, dis_z+1))
                    n6 = np.ravel_multi_index((i[0]+1, j[0], g[0]+1), (dis_rho+1, dis_phi+1, dis_z+1))
                    n7 = np.ravel_multi_index((i[0]+1, j[0]+1, g[0]+1), (dis_rho+1, dis_phi+1, dis_z+1))
                    n8 = np.ravel_multi_index((i[0], j[0]+1, g[0]+1), (dis_rho+1, dis_phi+1, dis_z+1))
                    
                    celdas.append([n1, n2, n3, n4, n5, n6, n7, n8])
                
                if((i[0]%dis_rho_vec==0)&(j[0]%dis_phi_vec==0)&(g[0]%dis_z_vec==0)):
                    Evec.append([np.sqrt(Ephi_m**2 + Erho_m**2)*np.cos(j[1]), np.sqrt(Ephi_m**2 + Erho_m**2)*np.sin(j[1]), 0])
                    Hvec.append([np.sqrt(Hrho_m**2 + Hphi_m**2 + Hz_m**2)*np.cos(j[1]), np.sqrt(Hrho_m**2 + Hphi_m**2 + Hz_m**2)*np.sin(j[1]), Hz_m])
                else:
                    Evec.append([0, 0, 0])
                    Hvec.append([0, 0, 0])

    malla = meshio.Mesh(puntos, {"hexahedron": celdas})

    malla.point_data['Campo_Electrico'] = E
    malla.point_data['Campo_Magnético'] = H

    malla = pv.wrap(malla)
    
    malla['Campo_Electrico_Vectorial'] = np.array(Evec)/np.max(Evec)
    malla['Campo_Magnético_Vectorial'] = np.array(Hvec)/np.max(Hvec)
    
    return malla

if __name__ == "__main__":
    
    n = 2
    m = 2
    l = 2

    a = 0.012
    d = 0.02

    mur = 1
    epsilonr = 1

    Aplus = 1

    dis_rho = 32
    dis_phi = 64
    dis_z = 32
    
    dis_rho_vec = 8
    dis_phi_vec = 4
    dis_z_vec = 3

    malla = TEnml_modes(n, m, l, a, d, mur, epsilonr, Aplus, dis_rho, dis_phi, dis_z, dis_rho_vec, dis_phi_vec, dis_z_vec)

    #malla.save("malla.vtk")

    plotter = pv.Plotter(shape=(1, 2))
    
    sel = 0 
    while((sel != 1)&(sel != 2)):
    
        print('Seleccione el campo que desea graficar:\n')
        print('1.   Campo eléctrico')
        print('2.   Campo magnético')
        
        sel = int(input())
    
    if (sel == 1):
        
        plotter.subplot(0, 0)
        plotter.add_mesh(malla, show_edges=False, cmap='jet', scalars='Campo_Electrico')

        plotter.subplot(0, 1)
        plotter.add_arrows(malla.points, malla['Campo_Electrico_Vectorial'], mag=0.004, cmap='jet')

    if (sel == 2):
        
        plotter.subplot(0, 0)
        plotter.add_mesh(malla, show_edges=False, cmap='jet', scalars='Campo_Magnético')

        plotter.subplot(0, 1)
        plotter.add_arrows(malla.points, malla['Campo_Magnético_Vectorial'], mag=0.004, cmap='jet')
        
    plotter.show_grid()
    plotter.show()