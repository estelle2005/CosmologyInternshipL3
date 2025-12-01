import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


H_0 = 73.2
Omega_m = 0.3
Omega_Lambda = 1 - Omega_m

### $w(a) = w_0 + (1-a)w_a $
### Compare $f(a)$ for :\n",


"### Hint:  then the equation for $H^\\prime$, then write a new function to solve the differential equation"


w_0_list = [-1.0, -1.2, -0.8, -1.0, -1.0]
w_a_list = [0, 0, 0, 0.1, -0.1]

plt.figure()

for i, W_0 in enumerate(w_0_list):
    W_a = w_a_list[i]

    def H(a):
        return H_0 *np.sqrt(Omega_m * a**-3 + Omega_Lambda*a**(-3*(1 + W_0 + W_a))*np.exp(3*W_a*(1-a)))

    """def H_prime(H, x):  #dérivée de H(x)
        #sans coef devant
        u_prime = Omega_m * np.exp(-3*x) + (1 + W_0 + W_a + np.exp(x))*Omega_Lambda* np.exp(-3*(x + W_0 * x + W_a * (x-1) + np.exp(x)))
        u = np.sqrt(Omega_m * np.exp(-3) + Omega_Lambda*np.exp(-3*(x + W_0*x + W_a*(x-1)+np.exp(x))))
        #on rajoute les coeffs
        return -H_0 * 3/2 * u_prime/u"""

    """def Omega_m_a(a):
        return Omega_m / (Omega_m + Omega_Lambda * a **-3)"""

    def dH_over_dlna(H, ln_a):
        #f' = df/dlna
        a = np.exp(ln_a)
        u_prime = Omega_m * a**(-3) + (1 + W_0 + W_a + a)*Omega_Lambda* a**(-3*(1 + W_0 + W_a ))*np.exp(3*(1-a))
        deriv = - H_0 * 3/2 * u_prime / H(a)
        return deriv

    #-- Definition of \"time\" = ln(a)
    a = 10.**np.linspace(-2, 0, 10000)  #de 10**-2 à 10**0
    ln_a = np.log(a)

    #-- Initial condition - pour z bien supérieur à 1 (2)
    f0 = 1
    f = odeint(dH_over_dlna, f0, ln_a)

    plt.plot(a, f, 
        linestyle='-', color=f'C{i}', linewidth=2, label=f'$\w_0$ = {W_0}; $\w_a$ = {W_a}')


plt.xlabel('Scale factor a')
plt.ylabel('Expansion rate H')
plt.xscale('log')
plt.grid(True)
plt.title('Expansion rate with evolving dark energy')
plt.legend()
plt.tight_layout()
plt.show()