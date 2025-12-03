import matplotlib.pyplot as plt
import fonctions
from scipy.stats import chisquare
import pandas as pd



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

r_d = 147.05 # Mpc today
c = 3 * 10**8

tableau = pd.read_csv('DESI_DR2_BAO_measurements.csv')

z = tableau['z_eff']

def Dv_over_rd(z):
    a = 1 / (1 + z)
    D_M = fonctions.d_A(z, pars) * (1+z)
    D_H = c / fonctions.H(a, pars)
    Dv = (z * D_M**2 * D_H)**(1/3)
    return Dv / r_d

def DM_over_DH(z):
    D_M = fonctions.d_A(z, pars) * (1+z)
    D_H = c / fonctions.H(a, pars)
    return D_M / D_H


def chi_carré():
    for z_i in z:

pars = {'Omega_m': ,'Omega_Lambda': ,'W_'}

iminuit
Least