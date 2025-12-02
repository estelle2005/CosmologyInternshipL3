import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


H_0 = 73.2

# $w(a) = w_0 + (1-a)w_a $

#-- Definition of \"time\" = ln(a)
a = 10.**np.linspace(-2, 0, 10000)  #de 10**-2 à 10**0
ln_a = np.log(a)


w_0_list = [-1, -1, -0.5, 0]
w_a_list = [0, 0, 0, 0]
Omega_Lambda_liste = [0.69, 0.72, 0.69, 0]


def H(a, pars):
    Omega_m = pars['Omega_m']
    W_0 = pars['W_0']
    W_a = pars['W_a']
    Omega_Lambda = 1 - Omega_m
    return H_0 *np.sqrt(Omega_m * a**-3 + Omega_Lambda*a**(-3*(1 + W_0 + W_a))*np.exp(-3*W_a*(1-a)))

def H_prime(a, pars): #on définit un dictionnaire pour les paramètres
    Omega_m = pars['Omega_m']
    W_0 = pars['W_0']
    W_a = pars['W_a']
    Omega_Lambda = 1 - Omega_m
    u_prime = Omega_m * a**(-3) + (1 + W_0 + W_a + a)*Omega_Lambda* a**(-3*(1 + W_0 + W_a))*np.exp(-3*W_a*(1-a))
    H_prime = - H_0 **2 * 3/2 * u_prime / H(a, pars)
    return H_prime

def Omega_m_a(a, pars):
    Omega_m = pars['Omega_m']
    W_0 = pars['W_0']
    W_a = pars['W_a']
    Omega_Lambda = 1 - Omega_m
    return Omega_m / (Omega_m + Omega_Lambda * a ** (-3*(W_0 + W_a)) * np.exp(-3*W_a*(1-a)))

def df_over_dlna(f, ln_a, pars):
    #f' = df/dlna
    a = np.exp(ln_a)
    #f' = - f² - \left( 2 + \frac{H\prime}{H} \right) f + \frac{3}{2}\Omega_m(a, w_a, w_0) 
    deriv = -f**2 - (2 + H_prime(a, pars)/H(a, pars))*f + 1.5*Omega_m_a(a, pars)
    return deriv

def growth_rate_f():
    f0 = 1 #condition initiale
    f = odeint(df_over_dlna, f0, ln_a)
    return f

def growth_factor_D():
    D_init = 0.01   #a_init = 0.01
    delta_lna = ln_a[1] - ln_a[0]
    term = growth_rate_f() * delta_lna
    int_dlnD = np.cumsum(term)
    ln_D = int_dlnD + np.log(D_init)
    D = np.exp(ln_D)
    return D

def plot_D():
    plt.figure() 
    for i, W_0 in enumerate(w_0_list):
        W_a = w_a_list[i]
        Omega_Lambda = Omega_Lambda_liste[i]
        Omega_m = 1 - Omega_Lambda
        plt.plot(a, growth_factor_D()/a, 
            linestyle='-', color=f'C{i}', linewidth=2, label=f'$W$ = {W_0}; $\Omega_\Lambda$ = {Omega_Lambda}')
    plt.xlabel('Scale factor a')
    plt.ylabel('Growth factor divided by a')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

