import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


H_0 = 73.2

# $w(a) = w_0 + (1-a)w_a $

w_0_list = [-1, -1, -0.5, 0]
w_a_list = [0, 0, 0, 0]
Omega_Lambda_liste = [0.69, 0.72, 0.69, 0]

plt.figure() 

for i, W_0 in enumerate(w_0_list):
    W_a = w_a_list[i]
    Omega_Lambda = Omega_Lambda_liste[i]
    Omega_m = 1 - Omega_Lambda

    def H(a):
        return H_0 *np.sqrt(Omega_m * a**-3 + Omega_Lambda*a**(-3*(1 + W_0 + W_a))*np.exp(-3*W_a*(1-a)))

    def H_prime(a):
        u_prime = Omega_m * a**(-3) + (1 + W_0 + W_a + a)*Omega_Lambda* a**(-3*(1 + W_0 + W_a))*np.exp(-3*W_a*(1-a))
        H_prime = - H_0 **2 * 3/2 * u_prime / H(a)
        return H_prime
    
    def Omega_m_a(a):
        return Omega_m / (Omega_m + Omega_Lambda * a ** (-3*(W_0 + W_a)) * np.exp(-3*W_a*(1-a)))

    def df_over_dlna(f, ln_a):
        #f' = df/dlna
        a = np.exp(ln_a)
        #f' = - f² - \left( 2 + \frac{H\prime}{H} \right) f + \frac{3}{2}\Omega_m(a, w_a, w_0) 
        deriv = -f**2 - (2 + H_prime(a)/H(a))*f + 1.5*Omega_m_a(a)
        return deriv

    #-- Definition of \"time\" = ln(a)Hi, I'm going to grab lunch at the cafeteria and join today's speaker. Would you be interested in join?
    a = 10.**np.linspace(-2, 0, 10000)  #de 10**-2 à 10**0
    ln_a = np.log(a)

    #-- Initial condition - pour z bien supérieur à 1 (2)
    f0 = 1
    f = odeint(df_over_dlna, f0, ln_a)

    a_init = 0.01
    D_init = 0.01

    delta_lna = ln_a[1] - ln_a[0]
    term = f * delta_lna
    int_dlnD = np.cumsum(term)

    ln_D = int_dlnD
    
    D = np.exp(ln_D)

    plt.plot(a, D/a, 
        linestyle='-', color=f'C{i}', linewidth=2, label=f'$W$ = {W_0}; $\Omega_\Lambda$ = {Omega_Lambda}')


plt.xlabel('Scale factor a')
plt.ylabel('Growth factor divided by a')
plt.xscale('log')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()