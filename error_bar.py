import matplotlib.pyplot as plt
import fonctions
from scipy.stats import chisquare


H_values = fonctions.H(a,pars)
a = 10.**np.linspace(-2, 0, 10000)  #de 10**-2 à 10**0
ln_a = np.log(a)
#pars = {'Omega_Lambda': Omega_Lambda, 'W_0': W_0_list[i], 'W_a': W_a_list[i]}


"""Exemple : 
# main.py
import fonctions

resultat = fonctions.addition(5, 3)
print(resultat)  # 8
print(fonctions.PI)  # 3.14159 import d'une variable globale
"""
"""# Effectifs observés et théoriques
observes = [16, 18, 22, 20, 24]
theoriques = [20, 20, 20, 20, 20]

# Test du khi carré d'adéquation
chi2, p = chisquare(observes, theoriques)

print(f"Statistique Khi carré : {chi2:.4f}")
print(f"p-value : {p:.4f}")"""
