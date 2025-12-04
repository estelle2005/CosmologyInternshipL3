import matplotlib.pyplot as plt
import fonctions
from scipy.stats import chisquare
import pandas as pd

import iminuit
from iminuit import minimize  # has same interface as scipy.optimize.minimize
from iminuit import Minuit, describe
from iminuit.cost import LeastSquares


"""H_values = fonctions.H(a,pars)
a = 10.**np.linspace(-2, 0, 10000)  #de 10**-2 à 10**0
ln_a = np.log(a)"""
#pars = {'Omega_Lambda': Omega_Lambda, 'W_0': W_0_list[i], 'W_a': W_a_list[i]}


"""Exemple : 
resultat = fonctions.addition(5, 3)
print(fonctions.PI)  # 3.14159 import d'une variable globale
"""
"""# Effectifs observés et théoriques
observes = [16, 18, 22, 20, 24]
theoriques = [20, 20, 20, 20, 20]

# Test du khi carré d'adéquation
chi2, p = chisquare(observes, theoriques)

print(f"Statistique Khi carré : {chi2:.4f}")
print(f"p-value : {p:.4f}")"""

'''def line(x, α, β):
    return α + x * β


# generate random toy data with random offsets in y
rng = np.random.default_rng(1)
data_x = np.linspace(0, 1, 10)
data_yerr = 0.1  # could also be an array
data_y = rng.normal(line(data_x, 1, 2), data_yerr)


least_squares = LeastSquares(data_x, data_y, data_yerr, line)

m = Minuit(least_squares, α=0, β=0)  # starting values for α and β

m.migrad()  # finds minimum of least_squares function
m.hesse()  # accurately computes uncertainties



# draw data and fitted line
plt.errorbar(data_x, data_y, data_yerr, fmt="ok", label="data")
plt.plot(data_x, line(data_x, *m.values), label="fit")

# display legend with some fit info
fit_info = [
    f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {m.fval:.1f} / {m.ndof:.0f} = {m.fmin.reduced_chi2:.1f}",
]
for p, v, e in zip(m.parameters, m.values, m.errors):
    fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")

plt.legend(title="\n".join(fit_info), frameon=False)
plt.xlabel("x")
plt.ylabel("y");

'''

r_d = 147.05 # Mpc today
c = 3 * 10**8

tableau = pd.read_csv('DESI_DR2_BAO_measurements.csv')
tableau = tableau.sort_values('z_eff').reset_index(drop=True)

z = tableau['z_eff']
DV_over_rd_exp = tableau['DV_over_rd']
sigma_DV_over_rd = tableau['DV_over_rd_err']
DM_over_DH_exp = tableau['DM_over_DH']
sigma_DM_over_DH = tableau['DM_over_DH_err']

def Dv_over_rd(z):
    a = 1 / (1 + z)
    D_M = fonctions.d_A(z, pars) * (1+z)
    D_H = c / fonctions.H(a, pars)
    Dv = (z * D_M**2 * D_H)**(1/3)
    return Dv / r_d

def DM_over_DH(z):
    a = 1 / (1+z)
    D_M = fonctions.d_A(z, pars) * (1+z)
    D_H = c / fonctions.H(a, pars)
    return D_M / D_H

def chi_carré_Dv_over_rd():
    sum = 0
    for i in range(len(z)):
        sum = sum + (DV_over_rd_exp[i] - Dv_over_rd(z[i]))/(sigma_DV_over_rd[i])**2
    return sum


def chi_carré_DM_over_DH():
    sum = 0
    for i in range(len(z)):
        sum = sum + (DM_over_DH_exp[i] - DM_over_DH(z[i]))/(sigma_DM_over_DH[i])**2


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


plot_Dv_over_rd_error_bar()

pars = {'Omega_m': 0.3,'Omega_Lambda': 0.7,'W_0': -1, 'W_a': 0}