import matplotlib.pyplot as plt
import numpy as np
import fonctions
from scipy.stats import chisquare
import pandas as pd


from iminuit import minimize  # has same interface as scipy.optimize.minimize
from iminuit import Minuit, describe
from iminuit.cost import LeastSquares


"""H_values = fonctions.H(a,pars)
a = 10.**np.linspace(-2, 0, 10000)  #de 10**-2 à 10**0
ln_a = np.log(a)"""
#pars = {'Omega_Lambda': Omega_Lambda, 'W_0': W_0_list[i], 'W_a': W_a_list[i]}


"""# Effectifs observés et théoriques
observes = [16, 18, 22, 20, 24]
theoriques = [20, 20, 20, 20, 20]

# Test du khi carré d'adéquation
chi2, p = chisquare(observes, theoriques)

print(f"Statistique Khi carré : {chi2:.4f}")
print(f"p-value : {p:.4f}")"""

r_d = 147.05 # Mpc today
c = 3 * 10**8

tableau = pd.read_csv('DESI_DR2_BAO_measurements.csv')
tableau = tableau.sort_values('z_eff').reset_index(drop=True)

z = tableau['z_eff']
DV_over_rd_exp = tableau['DV_over_rd']
sigma_DV_over_rd = tableau['DV_over_rd_err']
DM_over_DH_exp = tableau['DM_over_DH']
sigma_DM_over_DH = tableau['DM_over_DH_err']
 
#pars = {'Omega_m': 0.3,'Omega_Lambda': 0.7,'W_0': -1, 'W_a': 0}

def Dv_over_rd(z, pars):
    a = 1 / (1 + z)
    D_M = fonctions.d_A(z, pars) * (1+z)
    D_H = c / fonctions.H(a, pars)
    Dv = (z * D_M**2 * D_H)**(1/3)
    return Dv / r_d

def model_wrapper_Dv_over_rd(z, Omega_m, Omega_Lambda, W_0, W_a):
    pars = {'Omega_m': Omega_m,'Omega_Lambda': Omega_Lambda,'W_0': W_0, 'W_a': W_a}
    return Dv_over_rd(z, pars)

def iminuit_Dv_over_rd():
    cost = LeastSquares(z, DV_over_rd_exp, sigma_DV_over_rd, model_wrapper_Dv_over_rd)
    m = Minuit(cost, Omega_m=0.3, Omega_Lambda = 0.7, W_0 = -1, W_a = 0) 
    
    m.limits['Omega_m'] = (0.1, 1.0)
    m.limits['Omega_Lambda'] = (0.0, 1.0)
    m.limits['W_0'] = (-1.0, 0.0)
    m.limits['W_a'] = (-3.0, 0.0)
    
    m.migrad()  # finds minimum of least_squares function
    print("Résultat de l'ajustement:")
    print(f"$\Omega_m$ = {m.values['Omega_m']:.3f} ± {m.errors['Omega_m']:.3f}")
    print(f"$\Omega_\Lambda$= {m.values['Omega_Lambda']:.3f} ± {m.errors['Omega_Lambda']:.3f}")
    print(f"$w_0$ = {m.values['W_0']:.2f} ± {m.errors['W_0']:.2f}")
    print(f"$w_a$= {m.values['W_a']:.2f} ± {m.errors['W_a']:.2f}")
    print(f"χ²/dof = {m.fval:.2f}/{m.ndof} = {m.fval/m.ndof:.2f}")

    z_plot = np.linspace(min(z)-0.5, max(z)+0.5, 100)
    pars_fit = {
        'Omega_m': m.values['Omega_m'],
        'Omega_Lambda': m.values['Omega_Lambda'],
        'W_0': m.values['W_0'],
        'W_a': m.values['W_a']}
    DV_plot = np.array([Dv_over_rd(z, pars_fit) for z in z_plot])

    plt.figure()
    plt.errorbar(z, DV_over_rd_exp, yerr=sigma_DV_over_rd, fmt='o', capsize=5,
                label='Données BAO', color='darkblue')
    plt.plot(z_plot, DV_plot, 'r-', linewidth=2,
            label=f'Fit: $\Omega_m={m.values["Omega_m"]:.3f}, $\Omega_Lambda={m.values["Omega_Lambda"]:.3f},W_0={m.values["W_0"]:.2f}, W_a={m.values["W_a"]:.2f}')
    plt.xlabel('Redshift z')
    plt.ylabel(r'$D_V / r_d$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

iminuit_Dv_over_rd()

def DM_over_DH(z, pars):
    a = 1 / (1+z)
    D_M = fonctions.d_A(z, pars) * (1+z)
    D_H = c / fonctions.H(a, pars)
    return D_M / D_H

def model_wrapper_DM_over_DH(z, Omega_m, Omega_Lambda, W_0, W_a):
    pars = {'Omega_m': Omega_m,'Omega_Lambda': Omega_Lambda,'W_0': W_0, 'W_a': W_a}
    return DM_over_DH(z, pars)

def chi_carré_Dv_over_rd():
    sum = 0
    for i in range(len(z)):
        sum = sum + ((DV_over_rd_exp[i] - Dv_over_rd(z[i]))**2)/((sigma_DV_over_rd[i])**2)
    return sum

def chi_carré_DM_over_DH():
    sum = 0
    for i in range(len(z)):
        sum = sum + ((DM_over_DH_exp[i] - DM_over_DH(z[i]))**2)/((sigma_DM_over_DH[i])**2)
    return sum

def plot_Dv_over_rd_error_bar():
    plt.figure()
    plt.errorbar(z, DV_over_rd_exp, yerr=sigma_DV_over_rd, color='blue', 
             ecolor='red', label='Données ± erreur')
    plt.xlabel('$z$')
    plt.ylabel(r'$D_V / r_d$')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_DM_over_DH_error_bar():
    plt.figure()
    plt.errorbar(z, DM_over_DH_exp, yerr=sigma_DM_over_DH, color ='blue', 
             ecolor='red', label='Données ± erreur')
    plt.xlabel('$z$')
    plt.ylabel(r'$D_M / D_H$')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()