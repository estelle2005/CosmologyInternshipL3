import matplotlib.pyplot as plt
import numpy as np
import fonctions
import pandas as pd
from iminuit import Minuit
from iminuit.cost import LeastSquares

#plt.ion()

r_d = 147.05 # Mpc today
c = 3 * 10**5 # en km

tableau = pd.read_csv('DESI_DR2_BAO_measurements.csv')
tableau = tableau.sort_values('z_eff').reset_index(drop=True)

z_Dv = tableau['z_eff'].to_numpy()
DV_over_rd_exp = tableau['DV_over_rd'].to_numpy()
sigma_DV_over_rd = tableau['DV_over_rd_err'].to_numpy()
z_DM = tableau['z_eff'].to_numpy()[1:]
DM_over_DH_exp = tableau['DM_over_DH'].to_numpy()[1:]
sigma_DM_over_DH = tableau['DM_over_DH_err'].to_numpy()[1:]
 
#pars = {'Omega_m': 0.3,'Omega_Lambda': 0.7,'W_0': -1, 'W_a': 0, 'H_0': 73.2}

def Dv_over_rd(z_val, pars):
    a = 1 / (1 + z_val)
    D_A = fonctions.d_A(z_val, pars)
    D_M = D_A * (1+z_val)
    H_val = fonctions.H(a, pars)
    D_H = c / H_val
    Dv = (z_val * D_M**2 * D_H)**(1/3)
    return Dv / r_d

def model_wrapper_Dv_over_rd(z_val, Omega_m, W_0, W_a, H_0):
    pars = {'Omega_m': Omega_m,'Omega_Lambda': 1 - Omega_m,'W_0': W_0, 'W_a': W_a, 'H_0': H_0}
    return np.array([Dv_over_rd(z_i, pars) for z_i in z_val])

def iminuit_Dv_over_rd():
    cost = LeastSquares(z_Dv, DV_over_rd_exp, sigma_DV_over_rd, model_wrapper_Dv_over_rd)
    m = Minuit(cost, Omega_m=0.3, 
               #Omega_Lambda = 0.7,
               W_0 = -1, W_a = 0, H_0 = 73.2) 
    
    m.limits['Omega_m'] = (0.1, 1.0)
    #m.limits['Omega_Lambda'] = (0.0, 1.0)
    m.limits['W_0'] = (-2.0, 0.0)
    m.limits['W_a'] = (-3.0, 2.0)
    m.fixed['H_0'] = True

    m.migrad()  # finds minimum of least_squares function
    print("Résultat de l'ajustement:")
    print(f"$\Omega_m$ = {m.values['Omega_m']:.3f} ± {m.errors['Omega_m']:.3f}")
    #print(f"$\Omega_\Lambda$= {m.values['Omega_Lambda']:.3f} ± {m.errors['Omega_Lambda']:.3f}")
    print(f"$w_0$ = {m.values['W_0']:.2f} ± {m.errors['W_0']:.2f}")
    print(f"$w_a$= {m.values['W_a']:.2f} ± {m.errors['W_a']:.2f}")
    print(f"χ²      = {m.fval:.2f}")
    print(f"χ²/dof = {m.fval:.2f}/{m.ndof} = {m.fval/m.ndof:.2f}")

    z_plot = np.linspace(min(z_Dv)*0.9, max(z_Dv)*1.1, 200)
    pars_fit = {
        'Omega_m': m.values['Omega_m'],
        'Omega_Lambda': 1 - m.values['Omega_m'],
        'W_0': m.values['W_0'],
        'W_a': m.values['W_a'],
        'H_0': m.values['H_0']}
    
    DV_plot = np.array([Dv_over_rd(z_val, pars_fit) for z_val in z_plot])

    plt.figure()
    plt.errorbar(z_Dv, DV_over_rd_exp, yerr=sigma_DV_over_rd, fmt='o', capsize=5,
                label='Données BAO', color='darkblue')
    plt.plot(z_plot, DV_plot, 'r-', linewidth=2,
            label=f'Fit: $\Omega_m$={m.values["Omega_m"]:.3f}, $\Omega_\Lambda$= {pars_fit["Omega_Lambda"]:.3f},$w_0$={m.values["W_0"]:.2f}, $w_a$={m.values["W_a"]:.2f}')
    plt.xlabel('Redshift z')
    plt.ylabel(r'$D_V / r_d$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    return m, pars_fit


def DM_over_DH(z_val, pars):
    a = 1 / (1+z_val)
    D_M = fonctions.d_A(z_val, pars) * (1+z_val)
    D_H = c / fonctions.H(a, pars)
    return D_M / D_H

def model_wrapper_DM_over_DH(z_val, Omega_m, W_0, W_a, H_0):
    pars = {'Omega_m': Omega_m,'Omega_Lambda': 1 - Omega_m,'W_0': W_0, 'W_a': W_a, 'H_0': H_0}
    return np.array([DM_over_DH(z_i, pars) for z_i in z_val])

def iminuit_DM_over_DH():
    cost = LeastSquares(z, DM_over_DH_exp, sigma_DM_over_DH, model_wrapper_DM_over_DH)
    m = Minuit(cost, Omega_m=0.3, 
               #Omega_Lambda = 0.7,
               W_0 = -1, W_a = 0, H_0 = 73.2) 
    
    m.limits['Omega_m'] = (0.1, 1.0)
    #m.limits['Omega_Lambda'] = (0.0, 1.0)
    m.limits['W_0'] = (-2.0, 0.0)
    m.limits['W_a'] = (-3.0, 2.0)
    m.fixed['H_0'] = True

    m.migrad()  # finds minimum of least_squares function
    print("Résultat de l'ajustement:")
    print(f"$\Omega_m$ = {m.values['Omega_m']:.3f} ± {m.errors['Omega_m']:.3f}")
    #print(f"$\Omega_\Lambda$= {m.values['Omega_Lambda']:.3f} ± {m.errors['Omega_Lambda']:.3f}")
    print(f"$w_0$ = {m.values['W_0']:.2f} ± {m.errors['W_0']:.2f}")
    print(f"$w_a$= {m.values['W_a']:.2f} ± {m.errors['W_a']:.2f}")
    print(f"χ²      = {m.fval:.2f}")
    print(f"χ²/dof = {m.fval:.2f}/{m.ndof} = {m.fval/m.ndof:.2f}")

    z_plot = np.linspace(min(z_DM)*0.9, max(z_DM)*1.1, 200)
    pars_fit = {
        'Omega_m': m.values['Omega_m'],
        'Omega_Lambda': 1 - m.values['Omega_m'],
        'W_0': m.values['W_0'],
        'W_a': m.values['W_a'],
        'H_0': m.values['H_0']}
    
    DM_plot = np.array([DM_over_DH(z_val, pars_fit) for z_val in z_plot])

    plt.figure()
    plt.errorbar(z_DM, DM_over_DH_exp, yerr=sigma_DM_over_DH, fmt='o', capsize=5,
                label='Données BAO', color='darkblue')
    plt.plot(z_plot, DM_plot, 'r-', linewidth=2,
            label=f'Fit: $\Omega_m$={m.values["Omega_m"]:.3f}, $\Omega_\Lambda$= {pars_fit["Omega_Lambda"]:.3f},$w_0$={m.values["W_0"]:.2f}, $w_a$={m.values["W_a"]:.2f}')
    plt.xlabel('Redshift z')
    plt.ylabel(r'$D_M / D_H$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    return m, pars_fit
#ON PEUT METTRE LES PARAMÈTRES DU FIT DANS UN DICTIONNAIRE ET FAIRE UN PLOT SÉPARÉ

"""def chi_carré_Dv_over_rd(pars):
    sum = 0
    for i in range(len(z)):
        sum += ((DV_over_rd_exp[i] - Dv_over_rd(z[i], pars))**2)/((sigma_DV_over_rd[i])**2)
    return sum

def chi_carré_DM_over_DH(pars):
    sum = 0
    for i in range(len(z)):
        sum += ((DM_over_DH_exp[i] - DM_over_DH(z[i], pars))**2)/((sigma_DM_over_DH[i])**2)
    return sum"""

#COURBE THEORIQUE AVEC SET DE PARAMÈTRES FIXES ET DONNEES EXP AVEC BARRES D'ERREURS
def plot_Dv_over_rd_error_bar():
    plt.figure()
    pars = {'Omega_m': 0.3,'Omega_Lambda': 0.7,'W_0': -1, 'W_a': 0, 'H_0': 73.2}
    z_model = np.linspace(min(z_Dv)*0.9, max(z_Dv)*1.1, 200)
    f = [Dv_over_rd(z_i, pars) for z_i in z_model]
    plt.plot(z_model, f)
    plt.errorbar(z_Dv, DV_over_rd_exp, yerr=sigma_DV_over_rd, color='black', 
             ecolor='red', fmt='o', label='Données ± erreur')
    plt.xlabel('$z$')
    plt.ylabel(r'$D_V / r_d$')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_DM_over_DH_error_bar():
    plt.figure()
    pars = {'Omega_m': 0.3,'Omega_Lambda': 0.7,'W_0': -1, 'W_a': 0, 'H_0': 73.2}
    z_model = np.linspace(min(z_DM)*0.9, max(z_DM)*1.1, 200)
    f = [DM_over_DH(z_i, pars) for z_i in z_model]
    plt.plot(z_model, f)
    plt.errorbar(z_DM, DM_over_DH_exp, yerr=sigma_DM_over_DH, color ='black', 
             ecolor='red', fmt='o', label='Données ± erreur')
    plt.xlabel('$z$')
    plt.ylabel(r'$D_M / D_H$')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# 2 GRAPHIQUES - THEORIQUE PARAMÈTRES FIXES - BARRES D'ERREURS
def plot_Dv_over_rd_th():
    fig, axs = plt.subplots(nrows=2, ncols=1)
    pars = {'Omega_m': 0.3,'Omega_Lambda': 0.7,'W_0': -1, 'W_a': 0, 'H_0': 73.2}
    z_model = np.linspace(min(z_Dv)*0.9, max(z_Dv)*1.1, 200)
    f = [Dv_over_rd(z_i, pars) for z_i in z_model]
    axs[0].plot(z_model, f)
    axs[0].errorbar(z_Dv, DV_over_rd_exp, yerr=sigma_DV_over_rd, color='black', 
             ecolor='red', fmt='o', label='Données ± erreur')
    axs[0].set_xlabel('$z$')
    axs[0].set_ylabel(r'$D_V / r_d$')
    axs[0].grid(True)
    axs[0].legend()
    f_residu = [Dv_over_rd(z_i, pars) for z_i in z_Dv]
    residu = ((DV_over_rd_exp - f_residu)/sigma_DV_over_rd)
    axs[1].errorbar(z_Dv, residu, yerr=1, color='black', ecolor='red', fmt='o', label='Données ± erreur')
    axs[1].set_xlabel('$z$')
    axs[1].set_ylabel('Résidu normalisé')
    axs[1].grid(True)  
    axs[1].legend() 
    plt.show()

def plot_DM_over_DH_th():
    fig, axs = plt.subplots(nrows=2, ncols=1)
    pars = {'Omega_m': 0.3,'Omega_Lambda': 0.7,'W_0': -1, 'W_a': 0, 'H_0': 73.2}
    z_model = np.linspace(min(z_DM)*0.9, max(z_DM)*1.1, 200)
    f = [DM_over_DH(z_i, pars) for z_i in z_model]
    axs[0].plot(z_model, f)
    axs[0].errorbar(z_DM, DM_over_DH_exp, yerr=sigma_DM_over_DH, color ='black', 
             ecolor='red', fmt='o', label='Données ± erreur')
    axs[0].set_xlabel('$z$')
    axs[0].set_ylabel(r'$D_M / D_H$')
    axs[0].grid(True)
    axs[0].legend()
    f_residu = [DM_over_DH(z_i, pars) for z_i in z_DM]
    residu = ((DM_over_DH_exp - f_residu)/sigma_DM_over_DH)
    axs[1].errorbar(z_DM, residu, yerr=1, color='black', ecolor='red', fmt='o', label='Données ± erreur')
    axs[1].set_xlabel('$z$')
    axs[1].set_ylabel('Résidu normalisé')
    axs[1].grid(True)
    axs[1].legend()
    plt.show()


#2 GRAPHIQUES - AVEC FIT
def plot_fit_Dv_over_rd_error_bar():
    cost = LeastSquares(z_Dv, DV_over_rd_exp, sigma_DV_over_rd, model_wrapper_Dv_over_rd)
    m = Minuit(cost, Omega_m=0.3, 
               #Omega_Lambda = 0.7,
               W_0 = -1, W_a = 0, H_0 = 73.2) 
    
    m.limits['Omega_m'] = (0.1, 1.0)
    #m.limits['Omega_Lambda'] = (0.0, 1.0)
    m.limits['W_0'] = (-2.0, 0.0)
    m.limits['W_a'] = (-3.0, 2.0)
    m.fixed['H_0'] = True

    m.migrad()  # finds minimum of least_squares function
    print(m)
    print("Résultat de l'ajustement:")
    print(f"$\Omega_m$ = {m.values['Omega_m']:.3f} ± {m.errors['Omega_m']:.3f}")
    #print(f"$\Omega_\Lambda$= {m.values['Omega_Lambda']:.3f} ± {m.errors['Omega_Lambda']:.3f}")
    print(f"$w_0$ = {m.values['W_0']:.2f} ± {m.errors['W_0']:.2f}")
    print(f"$w_a$= {m.values['W_a']:.2f} ± {m.errors['W_a']:.2f}")
    print(f"χ²      = {m.fval:.2f}")
    print(f"χ²/dof = {m.fval:.2f}/{m.ndof} = {m.fval/m.ndof:.2f}")

    z_plot = np.linspace(min(z_Dv)*0.9, max(z_Dv)*1.1, 200)
    pars_fit = {
        'Omega_m': m.values['Omega_m'],
        'Omega_Lambda': 1 - m.values['Omega_m'],
        'W_0': m.values['W_0'],
        'W_a': m.values['W_a'],
        'H_0': m.values['H_0']}
    
    DV_plot = np.array([Dv_over_rd(z_val, pars_fit) for z_val in z_plot])

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize= (8,6))
    axs[0].errorbar(z_Dv, DV_over_rd_exp, yerr=sigma_DV_over_rd, fmt='o', capsize=5,
                label='BAO data', color='darkblue')
    axs[0].plot(z_plot, DV_plot, 'r-', linewidth=2,
            label=f'Fit: $\Omega_m$={m.values["Omega_m"]:.3f}, $\Omega_\Lambda$= {pars_fit["Omega_Lambda"]:.3f},$w_0$={m.values["W_0"]:.2f}, $w_a$={m.values["W_a"]:.2f}')
    axs[0].set_ylabel(r'$D_V / r_d$')
    #axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()
    f_residu = [Dv_over_rd(z_i, pars_fit) for z_i in z_Dv]
    residu = ((DV_over_rd_exp - f_residu)/sigma_DV_over_rd)
    axs[1].errorbar(z_Dv, residu, yerr=1, color='black', ecolor='red', fmt='o', label='Données ± erreur')
    axs[1].set_xlabel('$z$')
    axs[1].set_ylabel('Résidu normalisé')
    axs[1].grid(True)
    axs[1].legend()
    plt.savefig('/home/etudiant15/Documents/STAGE CPPM/Figures/Dv_over_rd_DESI_DR2_double.pdf', bbox_inches='tight')
    plt.show()
    return m, pars_fit

def plot_fit_DM_over_DH_error_bar():
    cost = LeastSquares(z_DM, DM_over_DH_exp, sigma_DM_over_DH, model_wrapper_DM_over_DH)
    m = Minuit(cost, Omega_m=0.3, 
               #Omega_Lambda = 0.7,
               W_0 = -1, W_a = 0, H_0 = 73.2) 
    
    m.limits['Omega_m'] = (0.1, 1.0)
    #m.limits['Omega_Lambda'] = (0.0, 1.0)
    m.limits['W_0'] = (-2.0, 0.0)
    m.limits['W_a'] = (-3.0, 2.0)
    """ m.fixed['W_0'] = True
    m.fixed['W_a'] = True"""
    m.fixed['H_0'] = True

    m.migrad()  # finds minimum of least_squares function
    print(m)
    print("Résultat de l'ajustement:")
    print(f"$\Omega_m$ = {m.values['Omega_m']:.3f} ± {m.errors['Omega_m']:.3f}")
    #print(f"$\Omega_\Lambda$= {m.values['Omega_Lambda']:.3f} ± {m.errors['Omega_Lambda']:.3f}")
    print(f"$w_0$ = {m.values['W_0']:.2f} ± {m.errors['W_0']:.2f}")
    print(f"$w_a$= {m.values['W_a']:.2f} ± {m.errors['W_a']:.2f}")
    print(f"χ²      = {m.fval:.2f}")
    print(f"χ²/dof = {m.fval:.2f}/{m.ndof} = {m.fval/m.ndof:.2f}")

    z_plot = np.linspace(min(z_DM)*0.9, max(z_DM)*1.1, 200)
    pars_fit = {
        'Omega_m': m.values['Omega_m'],
        'Omega_Lambda': 1 - m.values['Omega_m'],
        'W_0': m.values['W_0'],
        'W_a': m.values['W_a'],
        'H_0': m.values['H_0']}
    
    DM_plot = np.array([DM_over_DH(z_val, pars_fit) for z_val in z_plot])

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize= (8,6))
    axs[0].errorbar(z_DM, DM_over_DH_exp, yerr=sigma_DM_over_DH, fmt='o', capsize=5,
                label='Données BAO', color='darkblue')
    axs[0].plot(z_plot, DM_plot, 'r-', linewidth=2,
            label=f'Fit: $\Omega_m$={m.values["Omega_m"]:.3f}, $\Omega_\Lambda$= {pars_fit["Omega_Lambda"]:.3f},$w_0$={m.values["W_0"]:.2f}, $w_a$={m.values["W_a"]:.2f}')
    axs[0].set_ylabel(r'$D_M / D_H$')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()
    f_residu = [DM_over_DH(z_i, pars_fit) for z_i in z_DM]
    residu = ((DM_over_DH_exp - f_residu)/sigma_DM_over_DH)
    axs[1].errorbar(z_DM, residu, yerr=1, color='black', ecolor='red', fmt='o', label='Données ± erreur')
    axs[1].set_xlabel('$z$')
    axs[1].set_ylabel('Résidu normalisé')
    axs[1].grid(True)
    axs[1].legend()
    plt.savefig('/home/etudiant15/Documents/STAGE CPPM/Figures/DM_over_DH_DESI_DR2_double.pdf', bbox_inches='tight')
    plt.show()
    return m, pars_fit

plot_fit_DM_over_DH_error_bar()
#plot_fit_Dv_over_rd_error_bar()


KHI CARRE QUI SOMME 