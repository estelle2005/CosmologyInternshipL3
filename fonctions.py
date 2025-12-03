import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


H_0 = 73.2

# $W(a) = W_0 + (1-a)W_a $

#-- Definition of \"time\" = ln(a)
a = 10.**np.linspace(-2, 0, 10000)  #de 10**-2 à 10**0
ln_a = np.log(a)
z = 1/a - 1

def H(a, pars):
    Omega_Lambda = pars['Omega_Lambda']
    W_0 = pars['W_0']
    W_a = pars['W_a']
    Omega_m = 1 - Omega_Lambda
    return H_0 *np.sqrt(Omega_m * a**-3 + Omega_Lambda*a**(-3*(1 + W_0 + W_a))*np.exp(-3*W_a*(1-a)))

def H_prime(a, pars): #on définit un dictionnaire pour les paramètres
    Omega_Lambda = pars['Omega_Lambda']
    W_0 = pars['W_0']
    W_a = pars['W_a']
    Omega_m = 1 - Omega_Lambda
    u_prime = Omega_m * a**(-3) + (1 + W_0 + W_a + a)*Omega_Lambda* a**(-3*(1 + W_0 + W_a))*np.exp(-3*W_a*(1-a))
    H_prime = - H_0 **2 * 3/2 * u_prime / H(a, pars)
    return H_prime

def Omega_m_a(a, pars):
    Omega_Lambda = pars['Omega_Lambda']
    W_0 = pars['W_0']
    W_a = pars['W_a']
    Omega_m = 1 - Omega_Lambda
    return Omega_m / (Omega_m + Omega_Lambda * a ** (-3*(W_0 + W_a)) * np.exp(-3*W_a*(1-a)))

def df_over_dlna(f, ln_a, pars):
    #f' = df/dlna
    a = np.exp(ln_a)
    #f' = - f² - \left( 2 + \frac{H\prime}{H} \right) f + \frac{3}{2}\Omega_m(a, w_a, w_0) 
    deriv = -f**2 - (2 + H_prime(a, pars)/H(a, pars))*f + 1.5*Omega_m_a(a, pars)
    return deriv

def growth_rate_f(pars):
    f0 = 1 #condition initiale
    f = odeint(df_over_dlna, f0, ln_a, args=(pars,))
    return f

def growth_factor_D(pars):
    D_init = 0.01   #a_init = 0.01 - comme si on mettait 'A_s', cad on normalise
    delta_lna = ln_a[1] - ln_a[0]
    term = growth_rate_f(pars) * delta_lna
    int_dlnD = np.cumsum(term)
    ln_D = int_dlnD + np.log(D_init)
    D = np.exp(ln_D)
    return D


def plot_H_z_times_1plusz(): #derivée de a pour différentes valeurs de w_0 et w_a, Omega_Lambda fixé, en fonction de z
    plt.figure()
    W_0_list = [-1, -0.8, -0.6, -0.4, -0.2]
    W_a_list = [0, -0.6, -1.2, -1.8, -2.4]
    #Omega_m = 0.3
    Omega_Lambda = 0.7
    for i in range(len(W_a_list)):
        pars = {'Omega_Lambda': Omega_Lambda, 'W_0': W_0_list[i], 'W_a': W_a_list[i]}  
        plt.plot(z, H(a, pars) * a, 
            linestyle='-', color=f'C{i}', linewidth=2, label=f'$w_0$ = {W_0_list[i]}; $w_a$ = {W_a_list[i]}')
    plt.xlabel('Redshift z')
    plt.ylabel('$H(z)(1+z)[km/s/Mpc]$')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_D_over_a(): #D/a pour différentes valeurs de W et Omega_Lambda, en fonction de a
    plt.figure()
    W_0_list = [-1, -1, -0.5, 0]
    W_a_list = [0, 0, 0, 0]
    Omega_Lambda_list = [0.69, 0.72, 0.69, 0] 
    for i in range(len(Omega_Lambda_list)):
        pars = {'Omega_Lambda': Omega_Lambda_list[i], 'W_0': W_0_list[i], 'W_a': W_a_list[i]}  
        plt.plot(a, growth_factor_D(pars)/a, 
            linestyle='-', color=f'C{i}', linewidth=2, label=f'$W$ = {W_0_list[i]}; $\Omega_\Lambda$ = {Omega_Lambda_list[i]}')
    plt.xlabel('Scale factor a')
    plt.ylabel('Growth factor divided by a')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_D(): #D pour différentes valeurs de w_0 et w_a à Omega_Lambda fixé, en fonction de z
    plt.figure() # à vérifier
    W_0_list = [-1, -0.8, -0.6, -0.4, -0.2]
    W_a_list = [0, -0.6, -1.2, -1.8, -2.4]
    #Omega_m = 0.3
    Omega_Lambda = 0.7
    for i in range(len(W_a_list)):
        pars = {'Omega_Lambda': Omega_Lambda, 'W_0': W_0_list[i], 'W_a': W_a_list[i]}  
        plt.plot(z, growth_factor_D(pars), 
            linestyle='-', color=f'C{i}', linewidth=2, label=f'$w_0$ = {W_0_list[i]}; $w_a$ = {W_a_list[i]}')
    plt.xlabel('Redshift z')
    plt.ylabel('Growth factor $D_+$')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_f(): #f pour différentes valeurs de w_0 et w_a, en fonction de z, pour Omega_Lambda fixé
    plt.figure()
    W_0_list = [-1, -0.8, -0.6, -0.4, -0.2]
    W_a_list = [0, -0.6, -1.2, -1.8, -2.4]
    #Omega_m = 0.3
    Omega_Lambda = 0.7
    for i in range(len(W_a_list)):
        pars = {'Omega_Lambda': Omega_Lambda, 'W_0': W_0_list[i], 'W_a': W_a_list[i]}
        f_solution = growth_rate_f(pars)
        f_values = f_solution[:,0]
        plt.plot(z, f_values, 
            linestyle='-', color=f'C{i}', linewidth=2, label=f'$w_0$ = {W_0_list[i]}; $w_a$ = {W_a_list[i]}')
    plt.xlabel(f'$z$')
    plt.ylabel(f'$f(z)$')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def f_times_Dplus():
    plt.figure()
    W_0_list = [-1, -0.8, -0.6, -0.4, -0.2]
    W_a_list = [0, -0.6, -1.2, -1.8, -2.4]
    #Omega_m = 0.3
    Omega_Lambda = 0.7
    for i in range(len(W_a_list)):
        pars = {'Omega_Lambda': Omega_Lambda, 'W_0': W_0_list[i], 'W_a': W_a_list[i]}
        f_solution = growth_rate_f(pars)
        f_values = f_solution[:,0]
        plt.plot(z, f_values * growth_factor_D(pars), 
            linestyle='-', color=f'C{i}', linewidth=2, label=f'$w_0$ = {W_0_list[i]}; $w_a$ = {W_a_list[i]}')
    plt.xlabel('Redshift z')
    plt.ylabel(r'$f \times D_+$')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_f()
#plot_D()
#plot_H_z_times_1plusz()