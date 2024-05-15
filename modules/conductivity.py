import numpy as np
from scipy.special import jnp_zeros, jn
import pandas as pd

def material_conductivity(n, m, l, a, d, mur, epsilonr, Aplus, material, Q_un, Q_load0):
        
    c = 299792458

    mu0 = 4*np.pi*1E-7
    epsilon0 = 8.854E-12

    ceros_jnp = jnp_zeros(n, 10)

    pnm_p = ceros_jnp[m-1]

    f = (c/(2*np.pi*np.sqrt(mur*epsilonr)))*np.sqrt((pnm_p/a)**2 + (l*np.pi/d)**2)

    f_0 = material['f0']
    f_1 = material['f1']
    f_2 = material['f2']

    k = 2*np.pi*f*np.sqrt(mu0*mur*epsilon0*epsilonr)
    eta = np.sqrt((mu0*mur)/(epsilon0*epsilonr))

    beta = np.sqrt(k**2 - ((pnm_p/a)**2))
    
    H0 = -2*Aplus

    #Factor de calidad 
    Q_ex = 2/((1/Q_load0)-1/Q_un)
    Q_0 = 1/((Q_load0**(-1)) - (2*(Q_ex)**(-1)))
    Q_meas = f_0/(f_2-f_1)
    Q = 1/((Q_meas**(-1)) - (2*(Q_ex)**(-1)))

    #resistividad

    Rs1 = (((k**3)*eta*(a**4)*d*(1-(n**2)/(pnm_p**2))) / (4*(pnm_p**2)*Q_0)  )  *   (((((beta**2)*(a**4))/(pnm_p**2))  *  (1-((n**2)/(pnm_p**2))) + (0.5*a*d)*(1+((beta**2)*(a**2)*(n**2))/(pnm_p**4)))**(-1))
    Rs2 = (((k**3)*eta*(a**4)*d*(1-(n**2)/(pnm_p**2))) / (2*(pnm_p**2)*Q)  -   (Rs1*a*d*(1+((beta**2)*(a**2)*(n**2))/(pnm_p**4))))   *  ((pnm_p**2)/((beta**2)*(a**4)*(1-(n**2)/(pnm_p**2))))  - Rs1

    # Energías
    
    TSenergy = epsilon0*epsilonr*(k**2)*(eta**2)*(a**4)*(H0**2)*np.pi*d/(8*(pnm_p**2)) * (1-(n/pnm_p)**2)*jn(n, pnm_p)**2
    powerl = np.pi*(H0**2)*(jn(n,pnm_p)**2)/4  * (Rs1*d*a*(1+((beta**2)*(a**2)*(n**2))/(pnm_p**4))  +  ((Rs1+Rs2)*(beta**2)*(a**4))/(pnm_p**2)*(1-(n**2)/(pnm_p**2)))
    
    #conductividad

    sigma = (2*np.pi*f*mu0)/(2*(Rs2**2))
    
    print(f'Energía total almacenada (J) = {TSenergy}\n')
    print(f'Perdida de energía (J) = {powerl}\n')
    print(f'Factor de calidad medido = {Q_meas}\n')
    print(f'Factor de calidad Q0= {Q_0}\n')
    print(f'Factor de calidad = {Q}\n')
    print(f'Conductividad (MS/m) = {sigma*(1E-6)}\n')
    
    return TSenergy, powerl, Q_meas, Q_0, Q, sigma

if __name__ == "__main__":
    
    n = 1
    m = 1
    l = 1

    a = 0.012
    d = 0.02

    mur = 1
    epsilonr = 1
    
    Aplus = 1
    
    df = pd.read_json("data/measurements.json")
    
    material = df.loc['top cover','Copper']
    
    #material = {"f0": 10.47511340E9, "f1": 10.47236750E9, "f2": 10.47790310E9}
    
    Q_un = 7800
    Q_load0 = 2634.5
    
    TSenergy, powerl, Q_meas, Q_0, Q, sigma = material_conductivity(n, m, l, a, d, mur, epsilonr, Aplus, material, Q_un, Q_load0)