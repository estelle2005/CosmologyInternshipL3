import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constantes cosmologiques
H_0 = 73.2  # km/s/Mpc

omega_m_list = [0.1, 0.3, 0.7, 0.9]

plt.figure()

for i, Omega_m in enumerate(omega_m_list):
    Omega_Lambda = 1 - Omega_m

 # Equation : f \prime + f^2 + \left( 2 + \frac{H\prime}{H} \right) f - \frac{3}{2}\frac{\Omega_m}{\Omega_m + \Omega_\lambda a^3} = 0

    def H(a):
        return H_0 *np.sqrt(Omega_m * a**-3 + Omega_Lambda)

    def H_prime_over_H(a):
        #H'/H = dH/dlna /H = dlnH/dlna = -3/2*Omega_m(a)
        return -1.5 * Omega_m_a(a)

    def Omega_m_a(a):
        return Omega_m / (Omega_m + Omega_Lambda * a **3)

    def df_over_dlna(f, ln_a):
        #f' = df/dlna
        a = np.exp(ln_a)
        #f' = - f² - \left( 2 + \frac{H\prime}{H} \right) f + \frac{3}{2}\Omega_m(a) 
        deriv = -f**2 - (2 + H_prime_over_H(a))*f + 1.5*Omega_m_a(a)
        return deriv

    #-- Definition of \"time\" = ln(a)
    a = 10.**np.linspace(-2, 0, 10000)  #de 10**-2 à 10**0
    ln_a = np.log(a)

    #-- Initial condition - pour z bien supérieur à 1 (2)
    f0 = 1
    f = odeint(df_over_dlna, f0, ln_a)

    plt.plot(a, f, 
        linestyle='-', color=f'C{i}', linewidth=2, label=f'$\Omega_m$ = {Omega_m}; $\Omega_\lambda$ = {Omega_Lambda}')


plt.xlabel('Scale factor a')
plt.ylabel('Growth-rate f')
plt.xscale('log')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

"""Pour des valeurs de plus en plus petites de \Omega_m, le taux de croissance f diminue de plus en plus tôt (pour un facteur d'échelle de plus en plus faible)"""
