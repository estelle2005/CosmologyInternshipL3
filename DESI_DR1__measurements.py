import matplotlib.pyplot as plt
import numpy as np
import fonctions
import pandas as pd
from iminuit import Minuit
from iminuit.cost import LeastSquares


tableau = pd.read_csv('fs8_DESI_DR1_RSD+PV.csv')
tableau = tableau.sort_values('z_eff').reset_index(drop=True)

z = tableau['z_eff'].to_numpy()
fsigma8_exp = tableau['fsigma8'].to_numpy()
sigma_fsigma8 = tableau['fsigma8_err'].to_numpy()
 
#pars = {'Omega_m': 0.3,'Omega_Lambda': 0.7,'W_0': -1, 'W_a': 0, 'H_0': 73.2, 'sigma8': ...}

def fsigma8_th(z_val, pars):
    f_solution = fonctions.growth_rate_f(z_val, pars)
    f_values = f_solution
    return f_values * fonctions.growth_factor_D(z_val, pars) * pars['sigma8']

def model_wrapper_fsigma8(z_val, Omega_m, W_0, W_a, H_0, sigma8):
    pars = {'Omega_m': Omega_m,'Omega_Lambda': 1 - Omega_m,'W_0': W_0, 'W_a': W_a, 'H_0': H_0, 'sigma8': sigma8}
    return fsigma8_th(z_val, pars)


"""pars = {'Omega_m': Omega_m,'Omega_Lambda': 1 - Omega_m,'W_0': W_0, 'W_a': W_a, 'H_0': H_0}
    return [Dv_over_rd(z_i, pars) for z_i in z_val]"""

def iminuit_fsigma8():
    cost = LeastSquares(z, fsigma8_exp, sigma_fsigma8, model_wrapper_fsigma8)
    m = Minuit(cost, Omega_m=0.3, 
               #Omega_Lambda = 0.7,
               W_0 = -1, W_a = 0, H_0 = 73.2, sigma8 = 0.8) 
    
    m.limits['Omega_m'] = (0.1, 1.0)
    #m.limits['Omega_Lambda'] = (0.0, 1.0)
    m.limits['W_0'] = (-2.0, 0.0)
    m.limits['W_a'] = (-3.0, 2.0)
    m.fixed['H_0'] = True
    m.limits['sigma8'] = (0.6, 1.0)

    m.migrad()  # finds minimum of least_squares function
    print("Résultat de l'ajustement:")
    print(f"$\Omega_m$ = {m.values['Omega_m']:.3f} ± {m.errors['Omega_m']:.3f}")
    #print(f"$\Omega_\Lambda$= {m.values['Omega_Lambda']:.3f} ± {m.errors['Omega_Lambda']:.3f}")
    print(f"$w_0$ = {m.values['W_0']:.2f} ± {m.errors['W_0']:.2f}")
    print(f"$w_a$= {m.values['W_a']:.2f} ± {m.errors['W_a']:.2f}")
    print(f"$f_(\sigma 8)$ = {m.values['sigma8']:.2f} ± {m.errors['sigma8']:.2f}")
    print(f"χ²      = {m.fval:.2f}")
    print(f"χ²/dof = {m.fval:.2f}/{m.ndof} = {m.fval/m.ndof:.2f}")

    z_plot = np.linspace(min(z)*0.9, max(z)*1.1, 200)
    pars_fit = {
        'Omega_m': m.values['Omega_m'],
        'Omega_Lambda': 1 - m.values['Omega_m'],
        'W_0': m.values['W_0'],
        'W_a': m.values['W_a'],
        'H_0': m.values['H_0'],
        'sigma8': m.values['sigma8']
        }
    
    fsigma8_plot = np.array([fsigma8_th(z_val, pars_fit) for z_val in z_plot])

    plt.figure()
    plt.errorbar(z, fsigma8_exp, yerr=sigma_fsigma8, fmt='o', capsize=5,
                label='Données BAO', color='darkblue')
    plt.plot(z_plot, fsigma8_plot, 'r-', linewidth=2,
            label=f'Fit: $\Omega_m$={m.values["Omega_m"]:.3f}, $\Omega_\Lambda$= {pars_fit["Omega_Lambda"]:.3f},$w_0$={m.values["W_0"]:.2f}, $w_a$={m.values["W_a"]:.2f}, $f_\sigma8$={pars_fit["sigma8"]:.9f}')
    plt.xlabel('Redshift z')
    plt.ylabel(r'$f_{\sigma8}$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    return m, pars_fit

iminuit_fsigma8()




"""
def plot_DM_over_DH_error_bar():
    plt.figure()
    z_model = np.linspace(min(z)*0.9, max(z)*1.1, 200)
    f = [DM_over_DH(z_i, pars) for z_i in z_model]
    plt.plot(z_model, f)
    plt.errorbar(z, fsigma8_exp, yerr=fsigma8_err, color ='black', 
             ecolor='red', fmt='o', label='Données ± erreur')
    plt.xlabel('$z$')
    plt.ylabel(r'$D_M / D_H$')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    """