import numpy as np
from scipy.special import jn, yn, jnp_zeros

m = 1
n = 1
l = 1

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

f_0 = 10.476176000E9
f_1 = 10.474140250E9
f_2 = 10.478204250E9

f_0 = 10.47511100E9
f_1 = 10.47312230E9
f_2 = 10.47710870E9

f_0 = 10.47647500E9
f_1 = 10.47453550E9
f_2 = 10.47840900E9

f_0 = 10.47887560E9
f_1 = 10.47610680E9
f_2 = 10.48169800E9

f_0 = 10.47647500E9
f_1 = 10.47453550E9
f_2 = 10.47840900E9

k = 2*np.pi*f*np.sqrt(mu0*mur*epsilon0*epsilonr)
eta = np.sqrt((mu0*mur)/(epsilon0*epsilonr))

beta = np.sqrt(k**2 - ((pnm_p/a)**2))

#Factor de calidad 
Q_un = 7800
Q_ex = 7956.5
Q_load0 = 2634.5
Q_0 = 1/((Q_load0**(-1)) - (2*(Q_ex)**(-1)))
Q_meas = f_0/(f_2-f_1)
Q = 1/((Q_meas**(-1)) - (2*(Q_ex)**(-1)))


#resistividad

Rs1 = (((k**3)*eta*(a**4)*d*(1-(n**2)/(pnm_p**2))) / (4*(pnm_p**2)*Q_0)  )  *   (((((beta**2)*(a**4))/(pnm_p**2))  *  (1-((n**2)/(pnm_p**2))) + (0.5*a*d)*(1+((beta**2)*(a**2)*(n**2))/(pnm_p**4)))**(-1))
Rs2 = (((k**3)*eta*(a**4)*d*(1-(n**2)/(pnm_p**2))) / (2*(pnm_p**2)*Q)  -   (Rs1*a*d*(1+((beta**2)*(a**2)*(n**2))/(pnm_p**4))))   *  ((pnm_p**2)/((beta**2)*(a**4)*(1-(n**2)/(pnm_p**2))))  - Rs1

#conductividad

sigma = (2*np.pi*f*mu0)/(2*(Rs2**2))

print(f'Factor de calidad medido = {Q_meas}\n')
print(f'Factor de calidad = {Q}\n')
print(f'Factor de calidad Q0= {Q_0}\n')
print(f'Conductividad (MS/m) = {sigma*(1E-6)}')