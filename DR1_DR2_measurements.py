import matplotlib.pyplot as plt
import numpy as np
import fonctions, DESI_DR1__measurements as DR1, fit_DESI_DR2_BAO_measurements as DR2
import pandas as pd
from iminuit import Minuit
from iminuit.cost import LeastSquares
from matplotlib.lines import Line2D

r_d = 147.05 # Mpc today
c = 3 * 10**5 # en km

tableau_DR2 = pd.read_csv('DESI_DR2_BAO_measurements.csv')
tableau_DR2 = tableau_DR2.sort_values('z_eff').reset_index(drop=True)

z_Dv = tableau_DR2['z_eff'].to_numpy()
DV_over_rd_exp = tableau_DR2['DV_over_rd'].to_numpy()
sigma_DV_over_rd = tableau_DR2['DV_over_rd_err'].to_numpy()

z_DM = tableau_DR2['z_eff'].to_numpy()[1:]
DM_over_DH_exp = tableau_DR2['DM_over_DH'].to_numpy()[1:]
sigma_DM_over_DH = tableau_DR2['DM_over_DH_err'].to_numpy()[1:]


tableau_DR1 = pd.read_csv('fs8_DESI_DR1_RSD+PV.csv')
tableau_DR1 = tableau_DR1.sort_values('z_eff').reset_index(drop=True)

z = tableau_DR1['z_eff'].to_numpy()
fsigma8_exp = tableau_DR1['fsigma8'].to_numpy()
sigma_fsigma8 = tableau_DR1['fsigma8_err'].to_numpy()

#sans PV
tableau_DR1_noPV = pd.read_csv('fs8_DESI_DR1_RSDnoPV.csv')
tableau_DR1_noPV = tableau_DR1_noPV.sort_values('z_eff').reset_index(drop=True)

z_noPV = tableau_DR1_noPV['z_eff'].to_numpy()
fsigma8_exp_noPV = tableau_DR1_noPV['fsigma8'].to_numpy()
sigma_fsigma8_noPV = tableau_DR1_noPV['fsigma8_err'].to_numpy()

#H_0 fixé
"""def plot_fit_DR1_DR2():
    cost_f = LeastSquares(z, fsigma8_exp, sigma_fsigma8, DR1.model_wrapper_fsigma8) 
    cost_DM = LeastSquares(z_DM, DM_over_DH_exp, sigma_DM_over_DH, DR2.model_wrapper_DM_over_DH)
    cost_Dv = LeastSquares(z_Dv, DV_over_rd_exp, sigma_DV_over_rd, DR2.model_wrapper_Dv_over_rd)
    combined_cost = cost_f + cost_DM + cost_Dv
    m = Minuit(combined_cost, Omega_m=0.3, 
               #Omega_Lambda = 0.7,
               W_0 = -1, W_a = 0, H_0 = 73.2, sigma8 = 0.8) 
    
    m.limits['Omega_m'] = (0.1, 1.0)
    #m.limits['Omega_Lambda'] = (0.0, 1.0)
    m.limits['W_0'] = (-2.0, 0.0)
    m.limits['W_a'] = (-3.0, 2.0)
    m.fixed['H_0'] = True
    m.limits['sigma8'] = (0.6, 1.0)


    m.migrad()  # finds minimum of least_squares function
    m.minos()
    print(m)
    print("Résultat de l'ajustement:")
    print(f"$\Omega_m$ = {m.values['Omega_m']:.3f} ± {m.errors['Omega_m']:.3f}")
    #print(f"$\Omega_\Lambda$= {m.values['Omega_Lambda']:.3f} ± {m.errors['Omega_Lambda']:.3f}")
    print(f"$w_0$ = {m.values['W_0']:.2f} ± {m.errors['W_0']:.2f}")
    print(f"$w_a$= {m.values['W_a']:.2f} ± {m.errors['W_a']:.2f}")
    print(f"$H_0$= {m.values['H_0']:.2f} ± {m.errors['H_0']:.2f}")
    print(f"$f_(\sigma 8)$ = {m.values['sigma8']:.2f} ± {m.errors['sigma8']:.2f}")
    print(f"χ²      = {m.fval:.2f}")
    print(f"χ²/dof = {m.fval:.2f}/{m.ndof} = {m.fval/m.ndof:.2f}")

    pars_fit = {
        'Omega_m': m.values['Omega_m'],
        'Omega_Lambda': 1 - m.values['Omega_m'],
        'W_0': m.values['W_0'],
        'W_a': m.values['W_a'],
        'H_0': m.values['H_0'],
        'sigma8': m.values['sigma8']
        }
    
    chi2_f = np.sum(((fsigma8_exp - DR1.model_wrapper_fsigma8(z, *m.values)) / sigma_fsigma8)**2)       
    chi2_DM = np.sum(((DM_over_DH_exp - DR2.model_wrapper_DM_over_DH(z_DM, m.values['Omega_m'],
        m.values['W_0'], m.values['W_a'], m.values['H_0'])) / sigma_DM_over_DH)**2)
    chi2_DV = np.sum(((DV_over_rd_exp - DR2.model_wrapper_Dv_over_rd(z_Dv, m.values['Omega_m'],
        m.values['W_0'], m.values['W_a'], m.values['H_0'])) / sigma_DV_over_rd)**2)

    print(f"\nDétail des χ²:")
    print(f"  χ²_fsigma8 = {chi2_f:.2f}")
    print(f"  χ²_DM/DH = {chi2_DM:.2f}")
    print(f"  χ²_DV/rd = {chi2_DV:.2f}")
    print(f"  Somme vérifiée = {chi2_DM + chi2_DV:.2f}")

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize= (8,6))

    z_plot = np.linspace(min(z)*0.9, max(z)*1.1, 200)
    fsigma8_plot = np.array([DR1.fsigma8_th(z_val, pars_fit) for z_val in z_plot])
    axs[0].errorbar(z, fsigma8_exp, yerr=sigma_fsigma8, fmt='o', capsize=5,
                label='RSD Data', color='darkblue')
    axs[0].plot(z_plot, fsigma8_plot, 'r-', linewidth=2,
            label=f'Fit: $\Omega_m$={m.values["Omega_m"]:.3f}, $\Omega_\Lambda$= {pars_fit["Omega_Lambda"]:.3f},$w_0$={m.values["W_0"]:.2f}, $w_a$={m.values["W_a"]:.2f}, $f_\sigma8$={pars_fit["sigma8"]}')
    axs[0].set_xlabel('Redshift $z$')
    axs[0].set_ylabel(r'$f_{\sigma8}$')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()
    
    z_plot_DM = np.linspace(min(z_DM)*0.9, max(z_DM)*1.1, 200)
    DM_plot = np.array([DR2.DM_over_DH(z_val, pars_fit) for z_val in z_plot_DM])
    axs[1].errorbar(z_DM, DM_over_DH_exp, yerr=sigma_DM_over_DH, fmt='o', capsize=5,
                label='BAO Data', color='darkblue')
    axs[1].plot(z_plot_DM, DM_plot, 'r-', linewidth=2,
            label=f'Fit: $\Omega_m$={m.values["Omega_m"]:.3f}, $\Omega_\Lambda$= {pars_fit["Omega_Lambda"]:.3f},$w_0$={m.values["W_0"]:.2f}, $w_a$={m.values["W_a"]:.2f}')
    axs[1].set_ylabel(r'$D_M / D_H$')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    z_plot_DV = np.linspace(min(z_Dv)*0.9, max(z_Dv)*1.1, 200)
    DV_plot = np.array([DR2.Dv_over_rd(z_val, pars_fit) for z_val in z_plot_DV])
    axs[2].errorbar(z_Dv, DV_over_rd_exp, yerr=sigma_DV_over_rd, fmt='o', capsize=5,
                label='BAO Data', color='darkgreen')
    axs[2].plot(z_plot_DV, DV_plot, 'r-', linewidth=2,
            label=f'Fit: $\Omega_m$={m.values["Omega_m"]:.3f}, $\Omega_\Lambda$= {pars_fit["Omega_Lambda"]:.3f},$w_0$={m.values["W_0"]:.2f}, $w_a$={m.values["W_a"]:.2f}')
    axs[2].set_xlabel('Redshift $z$')
    axs[2].set_ylabel(r'$D_v / r_d$')
    axs[2].grid(True, alpha=0.3)
    axs[2].legend()

    plt.savefig('/home/etudiant15/Documents/STAGE CPPM/Figures/chi_3combined_DESI_DR1_DR2.pdf', bbox_inches='tight')
    plt.show()
    return m, pars_fit"""

#variable H_0 r_d
def plot_fit_DR1_DR2_PV():
    cost_f = LeastSquares(z, fsigma8_exp, sigma_fsigma8, DR1.model_wrapper_fsigma8) 
    cost_DM = LeastSquares(z_DM, DM_over_DH_exp, sigma_DM_over_DH, DR2.model_wrapper_DM_over_DH)
    cost_Dv = LeastSquares(z_Dv, DV_over_rd_exp, sigma_DV_over_rd, DR2.model_wrapper_Dv_over_rd)
    combined_cost = cost_f + cost_DM + cost_Dv
    m = Minuit(combined_cost, Omega_m=0.3, 
               #Omega_Lambda = 0.7,
               W_0 = -1, W_a = 0, H_0 = 73.2, sigma8 = 0.8, H_0xr_d = 10764.06) 
    
    m.limits['Omega_m'] = (0.1, 1.0)
    #m.limits['Omega_Lambda'] = (0.0, 1.0)
    m.limits['W_0'] = (-3.0, 1.0)
    m.limits['W_a'] = (-3.0, 2.0)
    m.fixed['H_0'] = True
    m.limits['sigma8'] = (0.6, 1.0)
    m.limits['H_0xr_d'] = (5000, 20000)


    m.migrad()
    m.minos()
    #m.draw_mncontour("W_0", "W_a", cl=(0.683, 0.954, 0.997), size=100)
    print(m)
    print("Résultat de l'ajustement:")
    print(f"$\Omega_m$ = {m.values['Omega_m']:.3f} ± {m.errors['Omega_m']:.3f}")
    #print(f"$\Omega_\Lambda$= {m.values['Omega_Lambda']:.3f} ± {m.errors['Omega_Lambda']:.3f}")
    print(f"$w_0$ = {m.values['W_0']:.2f} ± {m.errors['W_0']:.2f}")
    print(f"$w_a$= {m.values['W_a']:.2f} ± {m.errors['W_a']:.2f}")
    print(f"$H_0$= {m.values['H_0']:.2f} ± {m.errors['H_0']:.2f}")
    print(f"$H_0xr_d$= {m.values['H_0xr_d']:.2f} ± {m.errors['H_0xr_d']:.2f}")
    print(f"$f_(\sigma 8)$ = {m.values['sigma8']:.2f} ± {m.errors['sigma8']:.2f}")
    print(f"χ²      = {m.fval:.2f}")
    print(f"χ²/dof = {m.fval:.2f}/{m.ndof} = {m.fval/m.ndof:.2f}")

    pars_fit = {
        'Omega_m': m.values['Omega_m'],
        'Omega_Lambda': 1 - m.values['Omega_m'],
        'W_0': m.values['W_0'],
        'W_a': m.values['W_a'],
        'H_0': m.values['H_0'],
        'sigma8': m.values['sigma8'],
        'H_0xr_d': m.values['H_0xr_d']
        }
    
    """chi2_f = np.sum(((fsigma8_exp - DR1.fsigma8_th(z, m.values)) / sigma_fsigma8)**2)       
    chi2_DM = np.sum(((DM_over_DH_exp - DR2.model_wrapper_DM_over_DH(z_DM, m.values['Omega_m'],
        m.values['W_0'], m.values['W_a'], m.values['H_0'], m.values['H_0xr_d'])) / sigma_DM_over_DH)**2)
    chi2_DV = np.sum(((DV_over_rd_exp - DR2.model_wrapper_Dv_over_rd(z_Dv, m.values['Omega_m'],
        m.values['W_0'], m.values['W_a'], m.values['H_0'], m.values['H_0xr_d'])) / sigma_DV_over_rd)**2)

    print(f"\nDétail des χ²:")
    print(f"  χ²_fsigma8 = {chi2_f:.2f}")
    print(f"  χ²_DM/DH = {chi2_DM:.2f}")
    print(f"  χ²_DV/rd = {chi2_DV:.2f}")
    print(f"  Somme vérifiée = {chi2_DM + chi2_DV:.2f}")"""

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize= (8,8))

    param_line = plt.Line2D([0], [0], color='red', linewidth=2, label='Fit')

    z_plot = np.linspace(min(z)*0.9, max(z)*1.1, 200)
    fsigma8_plot = np.array([DR1.fsigma8_th(z_val, pars_fit) for z_val in z_plot])
    axs[0].errorbar(z, fsigma8_exp, yerr=sigma_fsigma8, fmt='o', capsize=5, color='darkblue')
    axs[0].plot(z_plot, fsigma8_plot, 'r-', linewidth=2)
    axs[0].set_ylabel(r'$f_{\sigma8}$')
    axs[0].grid(True, alpha=0.3)
    
    z_plot_DM = np.linspace(min(z_DM)*0.9, max(z_DM)*1.1, 200)
    DM_plot = np.array([DR2.DM_over_DH(z_val, pars_fit) for z_val in z_plot_DM])
    axs[1].errorbar(z_DM, DM_over_DH_exp, yerr=sigma_DM_over_DH, fmt='o', capsize=5, color='darkblue')
    axs[1].plot(z_plot_DM, DM_plot, 'r-', linewidth=2)
    axs[1].set_ylabel(r'$D_M / D_H$')
    axs[1].grid(True, alpha=0.3)
    
    z_plot_DV = np.linspace(min(z_Dv)*0.9, max(z_Dv)*1.1, 200)
    DV_plot = np.array([DR2.Dv_over_rd(z_val, pars_fit) for z_val in z_plot_DV])
    axs[2].errorbar(z_Dv, DV_over_rd_exp, yerr=sigma_DV_over_rd, fmt='o', capsize=5, color='darkgreen')
    axs[2].plot(z_plot_DV, DV_plot, 'r-', linewidth=2)
    axs[2].set_xlabel('Redshift $z$')
    axs[2].set_ylabel(r'$D_v / r_d$')
    axs[2].grid(True, alpha=0.3)
    
    param_text = (f'Fit: $\Omega_m={m.values["Omega_m"]:.3f}$, '
                  f'$\Omega_\Lambda={pars_fit["Omega_Lambda"]:.3f}$,\n'
                  f'$w_0={m.values["W_0"]:.2f}$, '
                  f'$w_a={m.values["W_a"]:.2f}$, '
                  f'$\sigma_8={pars_fit["sigma8"]:.2f}$, '
                  f'$H_0r_d={pars_fit["H_0xr_d"]:.2f}$')
    
    fig.text(0.5, 0.98, param_text, 
             ha='center', va='top', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))

    #fig.text(0.5, 0.99, param_text)
    plt.savefig('/home/etudiant15/Documents/STAGE CPPM/Figures/chi_3combined_DESI_BAO+RSD+PV.pdf', bbox_inches='tight')
    plt.show()

    merrors_m = m.merrors["Omega_m"]
    lower_m = merrors_m.lower
    upper_m = merrors_m.upper
    inf_m = - lower_m

    merrors_0 = m.merrors["W_0"]
    lower_0 = merrors_0.lower
    upper_0 = merrors_0.upper
    inf_0 = - lower_0

    merrors_a = m.merrors["W_a"]
    lower_a = merrors_a.lower
    upper_a = merrors_a.upper
    inf_a = - lower_a

    merrors_sigma = m.errors["sigma8"]
    lower_sigma = merrors_sigma
    upper_sigma = merrors_sigma
    inf_sigma = - lower_sigma

    merrors_H = m.merrors["H_0xr_d"]
    lower_H = merrors_H.lower
    upper_H = merrors_H.upper
    inf_H = - lower_H

    print(f'BAO+RSD+PV & ${m.values["Omega_m"]:.3f}^{{+{upper_m:.3f}}}_{{{- inf_m:.3f}}}$ & ${m.values["W_0"]:.3f}^{{+{upper_0:.3f}}}_{{{- inf_0:.3f}}}$ & ${m.values["W_a"]:.3f}^{{+{upper_a:.3f}}}_{{{- inf_a:.3f}}}$ & ${m.values["sigma8"]:.3f}^{{+{upper_sigma:.3f}}}_{{{inf_sigma:.3f}}}$ & ${m.values["H_0xr_d"]:.3f}^{{+{upper_H:.3f}}}_{{{- inf_H:.3f}}}$')
    return m, pars_fit

def plot_fit_DR1_DR2_noPV():
    cost_f = LeastSquares(z_noPV, fsigma8_exp_noPV, sigma_fsigma8_noPV, DR1.model_wrapper_fsigma8) 
    cost_DM = LeastSquares(z_DM, DM_over_DH_exp, sigma_DM_over_DH, DR2.model_wrapper_DM_over_DH)
    cost_Dv = LeastSquares(z_Dv, DV_over_rd_exp, sigma_DV_over_rd, DR2.model_wrapper_Dv_over_rd)
    combined_cost = cost_f + cost_DM + cost_Dv
    m = Minuit(combined_cost, Omega_m=0.3, 
               #Omega_Lambda = 0.7,
               W_0 = -1, W_a = 0, H_0 = 73.2, sigma8 = 0.8, H_0xr_d = 10764.06) 
    
    m.limits['Omega_m'] = (0.1, 1.0)
    #m.limits['Omega_Lambda'] = (0.0, 1.0)
    m.limits['W_0'] = (-3.0, 1.0)
    m.limits['W_a'] = (-3.0, 2.0)
    m.fixed['H_0'] = True
    m.limits['sigma8'] = (0.6, 1.0)
    m.limits['H_0xr_d'] = (3000, 30000)


    m.migrad()
    m.minos()
    #m.draw_mncontour("W_0", "W_a", cl=(0.683, 0.954,), size=100)
    print(m)
    print("Résultat de l'ajustement:")
    print(f"$\Omega_m$ = {m.values['Omega_m']:.3f} ± {m.errors['Omega_m']:.3f}")
    #print(f"$\Omega_\Lambda$= {m.values['Omega_Lambda']:.3f} ± {m.errors['Omega_Lambda']:.3f}")
    print(f"$w_0$ = {m.values['W_0']:.2f} ± {m.errors['W_0']:.2f}")
    print(f"$w_a$= {m.values['W_a']:.2f} ± {m.errors['W_a']:.2f}")
    print(f"$H_0$= {m.values['H_0']:.2f} ± {m.errors['H_0']:.2f}")
    print(f"$H_0xr_d$= {m.values['H_0xr_d']:.2f} ± {m.errors['H_0xr_d']:.2f}")
    print(f"$f_(\sigma 8)$ = {m.values['sigma8']:.2f} ± {m.errors['sigma8']:.2f}")
    print(f"χ²      = {m.fval:.2f}")
    print(f"χ²/dof = {m.fval:.2f}/{m.ndof} = {m.fval/m.ndof:.2f}")

    pars_fit = {
        'Omega_m': m.values['Omega_m'],
        'Omega_Lambda': 1 - m.values['Omega_m'],
        'W_0': m.values['W_0'],
        'W_a': m.values['W_a'],
        'H_0': m.values['H_0'],
        'sigma8': m.values['sigma8'],
        'H_0xr_d': m.values['H_0xr_d']
        }
    
    chi2_f = np.sum(((fsigma8_exp_noPV - DR1.fsigma8_th(z_noPV, m.values)) / sigma_fsigma8_noPV)**2)       
    chi2_DM = np.sum(((DM_over_DH_exp - DR2.model_wrapper_DM_over_DH(z_DM, m.values['Omega_m'],
        m.values['W_0'], m.values['W_a'], m.values['H_0'], m.values['H_0xr_d'])) / sigma_DM_over_DH)**2)
    chi2_DV = np.sum(((DV_over_rd_exp - DR2.model_wrapper_Dv_over_rd(z_Dv, m.values['Omega_m'],
        m.values['W_0'], m.values['W_a'], m.values['H_0'], m.values['H_0xr_d'])) / sigma_DV_over_rd)**2)

    print(f"\nDétail des χ²:")
    print(f"  χ²_fsigma8 = {chi2_f:.2f}")
    print(f"  χ²_DM/DH = {chi2_DM:.2f}")
    print(f"  χ²_DV/rd = {chi2_DV:.2f}")
    print(f"  Somme vérifiée = {chi2_DM + chi2_DV:.2f}")

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize= (8,8))

    param_line = plt.Line2D([0], [0], color='red', linewidth=2, label='Fit')

    z_plot = np.linspace(min(z)*0.9, max(z)*1.1, 200)
    fsigma8_plot = np.array([DR1.fsigma8_th(z_val, pars_fit) for z_val in z_plot])
    axs[0].errorbar(z_noPV, fsigma8_exp_noPV, yerr=sigma_fsigma8_noPV, fmt='o', capsize=5, color='darkblue')
    axs[0].plot(z_plot, fsigma8_plot, 'r-', linewidth=2)
    axs[0].set_ylabel(r'$f_{\sigma8}$')
    axs[0].grid(True, alpha=0.3)
    
    z_plot_DM = np.linspace(min(z_DM)*0.9, max(z_DM)*1.1, 200)
    DM_plot = np.array([DR2.DM_over_DH(z_val, pars_fit) for z_val in z_plot_DM])
    axs[1].errorbar(z_DM, DM_over_DH_exp, yerr=sigma_DM_over_DH, fmt='o', capsize=5, color='darkblue')
    axs[1].plot(z_plot_DM, DM_plot, 'r-', linewidth=2)
    axs[1].set_ylabel(r'$D_M / D_H$')
    axs[1].grid(True, alpha=0.3)
    
    z_plot_DV = np.linspace(min(z_Dv)*0.9, max(z_Dv)*1.1, 200)
    DV_plot = np.array([DR2.Dv_over_rd(z_val, pars_fit) for z_val in z_plot_DV])
    axs[2].errorbar(z_Dv, DV_over_rd_exp, yerr=sigma_DV_over_rd, fmt='o', capsize=5, color='darkgreen')
    axs[2].plot(z_plot_DV, DV_plot, 'r-', linewidth=2)
    axs[2].set_xlabel('Redshift $z$')
    axs[2].set_ylabel(r'$D_v / r_d$')
    axs[2].grid(True, alpha=0.3)
    
    param_text = (f'Fit: $\Omega_m={m.values["Omega_m"]:.3f}$, '
                  f'$\Omega_\Lambda={pars_fit["Omega_Lambda"]:.3f}$,\n'
                  f'$w_0={m.values["W_0"]:.2f}$, '
                  f'$w_a={m.values["W_a"]:.2f}$, '
                  f'$\sigma_8={pars_fit["sigma8"]:.2f}$, '
                  f'$H_0r_d={pars_fit["H_0xr_d"]:.2f}$')
    
    fig.text(0.5, 0.98, param_text, 
             ha='center', va='top', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))

    plt.savefig('/home/etudiant15/Documents/STAGE CPPM/Figures/chi_3combined_DESI_BAO+RSD.pdf', bbox_inches='tight')
    plt.show()

    merrors_m = m.merrors["Omega_m"]
    lower_m = merrors_m.lower
    upper_m = merrors_m.upper
    inf_m = - lower_m

    merrors_0 = m.merrors["W_0"]
    lower_0 = merrors_0.lower
    upper_0 = merrors_0.upper
    inf_0 = - lower_0

    merrors_a = m.merrors["W_a"]
    lower_a = merrors_a.lower
    upper_a = merrors_a.upper
    inf_a = - lower_a

    merrors_sigma = m.errors["sigma8"]
    lower_sigma = merrors_sigma
    upper_sigma = merrors_sigma
    inf_sigma = - lower_sigma

    merrors_H = m.merrors["H_0xr_d"]
    lower_H = merrors_H.lower
    upper_H = merrors_H.upper
    inf_H = - lower_H

    print(f'BAO+RSD & ${m.values["Omega_m"]:.3f}^{{+{upper_m:.3f}}}_{{{- inf_m:.3f}}}$ & ${m.values["W_0"]:.3f}^{{+{upper_0:.3f}}}_{{{- inf_0:.3f}}}$ & ${m.values["W_a"]:.3f}^{{+{upper_a:.3f}}}_{{{- inf_a:.3f}}}$ & ${m.values["sigma8"]:.3f}^{{+{upper_sigma:.3f}}}_{{{inf_sigma:.3f}}}$ & ${m.values["H_0xr_d"]:.3f}^{{+{upper_H:.3f}}}_{{{- inf_H:.3f}}}$')
    return m, pars_fit

#wCDM - 
def plot_fit_DR1_DR2_PV_wCDM():
    cost_f = LeastSquares(z, fsigma8_exp, sigma_fsigma8, DR1.model_wrapper_fsigma8) 
    cost_DM = LeastSquares(z_DM, DM_over_DH_exp, sigma_DM_over_DH, DR2.model_wrapper_DM_over_DH)
    cost_Dv = LeastSquares(z_Dv, DV_over_rd_exp, sigma_DV_over_rd, DR2.model_wrapper_Dv_over_rd)
    combined_cost = cost_f + cost_DM + cost_Dv
    m = Minuit(combined_cost, Omega_m=0.3, 
               #Omega_Lambda = 0.7,
               W_0 = -1, W_a = 0, H_0 = 73.2, sigma8 = 0.8, H_0xr_d = 10764.06) 
    
    m.limits['Omega_m'] = (0.1, 1.0)
    #m.limits['Omega_Lambda'] = (0.0, 1.0)
    m.limits['W_0'] = (-3.0, 1.0)
    m.fixed['W_a'] = True
    m.fixed['H_0'] = True
    m.limits['sigma8'] = (0.6, 1.0)
    m.limits['H_0xr_d'] = (3000, 30000)


    m.migrad()
    m.minos()
    m.draw_mncontour("Omega_m", "W_0", cl=(0.683, 0.954, ), size=100)
    #plt.xlim(0.2, 0.4)
    #plt.ylim(-1.25, -0.5)
    plt.axhline(-1, color='black', ls=':')
    plt.savefig('/home/etudiant15/Documents/STAGE CPPM/Figures/mncontour_BAO+RSD+PV_wCDM.pdf', bbox_inches='tight')
    
    print(m)
    print("Résultat de l'ajustement:")
    print(f"$\Omega_m$ = {m.values['Omega_m']:.3f} ± {m.errors['Omega_m']:.3f}")
    #print(f"$\Omega_\Lambda$= {m.values['Omega_Lambda']:.3f} ± {m.errors['Omega_Lambda']:.3f}")
    print(f"$w_0$ = {m.values['W_0']:.2f} ± {m.errors['W_0']:.2f}")
    print(f"$w_a$= {m.values['W_a']:.2f} ± {m.errors['W_a']:.2f}")
    print(f"$H_0$= {m.values['H_0']:.2f} ± {m.errors['H_0']:.2f}")
    print(f"$H_0xr_d$= {m.values['H_0xr_d']:.2f} ± {m.errors['H_0xr_d']:.2f}")
    print(f"$f_(\sigma 8)$ = {m.values['sigma8']:.2f} ± {m.errors['sigma8']:.2f}")
    print(f"χ²      = {m.fval:.2f}")
    print(f"χ²/dof = {m.fval:.2f}/{m.ndof} = {m.fval/m.ndof:.2f}")

    pars_fit = {
        'Omega_m': m.values['Omega_m'],
        'Omega_Lambda': 1 - m.values['Omega_m'],
        'W_0': m.values['W_0'],
        'W_a': m.values['W_a'],
        'H_0': m.values['H_0'],
        'sigma8': m.values['sigma8'],
        'H_0xr_d': m.values['H_0xr_d']
        }
    
    chi2_f = np.sum(((fsigma8_exp - DR1.fsigma8_th(z, m.values)) / sigma_fsigma8)**2)       
    chi2_DM = np.sum(((DM_over_DH_exp - DR2.model_wrapper_DM_over_DH(z_DM, m.values['Omega_m'],
        m.values['W_0'], m.values['W_a'], m.values['H_0'], m.values['H_0xr_d'])) / sigma_DM_over_DH)**2)
    chi2_DV = np.sum(((DV_over_rd_exp - DR2.model_wrapper_Dv_over_rd(z_Dv, m.values['Omega_m'],
        m.values['W_0'], m.values['W_a'], m.values['H_0'], m.values['H_0xr_d'])) / sigma_DV_over_rd)**2)

    print(f"\nDétail des χ²:")
    print(f"  χ²_fsigma8 = {chi2_f:.2f}")
    print(f"  χ²_DM/DH = {chi2_DM:.2f}")
    print(f"  χ²_DV/rd = {chi2_DV:.2f}")
    print(f"  Somme vérifiée = {chi2_DM + chi2_DV + chi2_f:.2f}")

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize= (8,8))

    param_line = plt.Line2D([0], [0], color='red', linewidth=2, label='Fit')

    z_plot = np.linspace(min(z)*0.9, max(z)*1.1, 200)
    fsigma8_plot = np.array([DR1.fsigma8_th(z_val, pars_fit) for z_val in z_plot])
    axs[0].errorbar(z, fsigma8_exp, yerr=sigma_fsigma8, fmt='o', capsize=5, color='darkblue')
    axs[0].plot(z_plot, fsigma8_plot, 'r-', linewidth=2)
    axs[0].set_ylabel(r'$f_{\sigma8}$')
    axs[0].grid(True, alpha=0.3)
    
    z_plot_DM = np.linspace(min(z_DM)*0.9, max(z_DM)*1.1, 200)
    DM_plot = np.array([DR2.DM_over_DH(z_val, pars_fit) for z_val in z_plot_DM])
    axs[1].errorbar(z_DM, DM_over_DH_exp, yerr=sigma_DM_over_DH, fmt='o', capsize=5, color='darkblue')
    axs[1].plot(z_plot_DM, DM_plot, 'r-', linewidth=2)
    axs[1].set_ylabel(r'$D_M / D_H$')
    axs[1].grid(True, alpha=0.3)
    
    z_plot_DV = np.linspace(min(z_Dv)*0.9, max(z_Dv)*1.1, 200)
    DV_plot = np.array([DR2.Dv_over_rd(z_val, pars_fit) for z_val in z_plot_DV])
    axs[2].errorbar(z_Dv, DV_over_rd_exp, yerr=sigma_DV_over_rd, fmt='o', capsize=5, color='darkgreen')
    axs[2].plot(z_plot_DV, DV_plot, 'r-', linewidth=2)
    axs[2].set_xlabel('Redshift $z$')
    axs[2].set_ylabel(r'$D_v / r_d$')
    axs[2].grid(True, alpha=0.3)
    
    param_text = (f'Fit: $\Omega_m={m.values["Omega_m"]:.3f}$, '
                  f'$\Omega_\Lambda={pars_fit["Omega_Lambda"]:.3f}$,\n'
                  f'$w_0={m.values["W_0"]:.2f}$, '
                  f'$w_a={m.values["W_a"]:.2f}$, '
                  f'$\sigma_8={pars_fit["sigma8"]:.2f}$, '
                  f'$H_0r_d={pars_fit["H_0xr_d"]:.2f}$')
    
    fig.text(0.5, 0.98, param_text, 
             ha='center', va='top', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))

    #fig.text(0.5, 0.99, param_text)
    plt.savefig('/home/etudiant15/Documents/STAGE CPPM/Figures/chi_3combined_DESI_BAO+RSD+PV_wCDM.pdf', bbox_inches='tight')
    plt.show()

    merrors_m = m.merrors["Omega_m"]
    lower_m = merrors_m.lower
    upper_m = merrors_m.upper
    inf_m = - lower_m

    merrors_0 = m.merrors["W_0"]
    lower_0 = merrors_0.lower
    upper_0 = merrors_0.upper
    inf_0 = - lower_0

    #merrors_a = m.merrors["W_a"]
    #lower_a = merrors_a.lower
    #upper_a = merrors_a.upper
    #inf_a = - lower_a

    merrors_sigma = m.errors["sigma8"]
    lower_sigma = merrors_sigma
    upper_sigma = merrors_sigma
    inf_sigma = - lower_sigma

    merrors_H = m.merrors["H_0xr_d"]
    lower_H = merrors_H.lower
    upper_H = merrors_H.upper
    inf_H = - lower_H

    print(f'BAO+RSD+PV & '+
        f'${m.values["Omega_m"]:.3f}^{{+{upper_m:.3f}}}_{{{- inf_m:.3f}}}$ & '+
        f'${m.values["W_0"]:.3f}^{{+{upper_0:.3f}}}_{{{- inf_0:.3f}}}$ & 0 & '+
        f'${m.values["sigma8"]:.3f}^{{+{upper_sigma:.3f}}}_{{{inf_sigma:.3f}}}$ & '+
        f'${m.values["H_0xr_d"]:.3f}^{{+{upper_H:.3f}}}_{{{- inf_H:.3f}}}$')
    return m, pars_fit

def plot_fit_DR1_DR2_noPV_wCDM():
    cost_f = LeastSquares(z_noPV, fsigma8_exp_noPV, sigma_fsigma8_noPV, DR1.model_wrapper_fsigma8) 
    cost_DM = LeastSquares(z_DM, DM_over_DH_exp, sigma_DM_over_DH, DR2.model_wrapper_DM_over_DH)
    cost_Dv = LeastSquares(z_Dv, DV_over_rd_exp, sigma_DV_over_rd, DR2.model_wrapper_Dv_over_rd)
    combined_cost = cost_f + cost_DM + cost_Dv
    m = Minuit(combined_cost, Omega_m=0.3, 
               #Omega_Lambda = 0.7,
               W_0 = -1, W_a = 0, H_0 = 73.2, sigma8 = 0.8, H_0xr_d = 10764.06) 
    
    m.limits['Omega_m'] = (0.1, 1.0)
    #m.limits['Omega_Lambda'] = (0.0, 1.0)
    m.limits['W_0'] = (-3.0, 1.0)
    m.fixed['W_a'] = True
    m.fixed['H_0'] = True
    m.limits['sigma8'] = (0.6, 1.0)
    m.limits['H_0xr_d'] = (3000, 30000)


    m.migrad()
    m.minos()
    m.draw_mncontour("Omega_m", "W_0", cl=(0.683, 0.954, ), size=100)
    #plt.xlim(0.2, 0.4)
    #plt.ylim(-1.25, -0.5)
    plt.axhline(-1, color='black', ls=':')
    plt.savefig('/home/etudiant15/Documents/STAGE CPPM/Figures/mncontour_BAO+RSD_wCDM.pdf', bbox_inches='tight')
    
    print(m)
    print("Résultat de l'ajustement:")
    print(f"$\Omega_m$ = {m.values['Omega_m']:.3f} ± {m.errors['Omega_m']:.3f}")
    #print(f"$\Omega_\Lambda$= {m.values['Omega_Lambda']:.3f} ± {m.errors['Omega_Lambda']:.3f}")
    print(f"$w_0$ = {m.values['W_0']:.2f} ± {m.errors['W_0']:.2f}")
    print(f"$w_a$= {m.values['W_a']:.2f} ± {m.errors['W_a']:.2f}")
    print(f"$H_0$= {m.values['H_0']:.2f} ± {m.errors['H_0']:.2f}")
    print(f"$H_0xr_d$= {m.values['H_0xr_d']:.2f} ± {m.errors['H_0xr_d']:.2f}")
    print(f"$f_(\sigma 8)$ = {m.values['sigma8']:.2f} ± {m.errors['sigma8']:.2f}")
    print(f"χ²      = {m.fval:.2f}")
    print(f"χ²/dof = {m.fval:.2f}/{m.ndof} = {m.fval/m.ndof:.2f}")

    pars_fit = {
        'Omega_m': m.values['Omega_m'],
        'Omega_Lambda': 1 - m.values['Omega_m'],
        'W_0': m.values['W_0'],
        'W_a': m.values['W_a'],
        'H_0': m.values['H_0'],
        'sigma8': m.values['sigma8'],
        'H_0xr_d': m.values['H_0xr_d']
        }
    
    chi2_f = np.sum(((fsigma8_exp_noPV - DR1.fsigma8_th(z_noPV, m.values)) / sigma_fsigma8_noPV)**2)       
    chi2_DM = np.sum(((DM_over_DH_exp - DR2.model_wrapper_DM_over_DH(z_DM, m.values['Omega_m'],
        m.values['W_0'], m.values['W_a'], m.values['H_0'], m.values['H_0xr_d'])) / sigma_DM_over_DH)**2)
    chi2_DV = np.sum(((DV_over_rd_exp - DR2.model_wrapper_Dv_over_rd(z_Dv, m.values['Omega_m'],
        m.values['W_0'], m.values['W_a'], m.values['H_0'], m.values['H_0xr_d'])) / sigma_DV_over_rd)**2)

    print(f"\nDétail des χ²:")
    print(f"  χ²_fsigma8 = {chi2_f:.2f}")
    print(f"  χ²_DM/DH = {chi2_DM:.2f}")
    print(f"  χ²_DV/rd = {chi2_DV:.2f}")
    print(f"  Somme vérifiée = {chi2_DM + chi2_DV:.2f}")

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize= (8,8))

    param_line = plt.Line2D([0], [0], color='red', linewidth=2, label='Fit')

    z_plot = np.linspace(min(z)*0.9, max(z)*1.1, 200)
    fsigma8_plot = np.array([DR1.fsigma8_th(z_val, pars_fit) for z_val in z_plot])
    axs[0].errorbar(z_noPV, fsigma8_exp_noPV, yerr=sigma_fsigma8_noPV, fmt='o', capsize=5, color='darkblue')
    axs[0].plot(z_plot, fsigma8_plot, 'r-', linewidth=2)
    axs[0].set_ylabel(r'$f_{\sigma8}$')
    axs[0].grid(True, alpha=0.3)
    
    z_plot_DM = np.linspace(min(z_DM)*0.9, max(z_DM)*1.1, 200)
    DM_plot = np.array([DR2.DM_over_DH(z_val, pars_fit) for z_val in z_plot_DM])
    axs[1].errorbar(z_DM, DM_over_DH_exp, yerr=sigma_DM_over_DH, fmt='o', capsize=5, color='darkblue')
    axs[1].plot(z_plot_DM, DM_plot, 'r-', linewidth=2)
    axs[1].set_ylabel(r'$D_M / D_H$')
    axs[1].grid(True, alpha=0.3)
    
    z_plot_DV = np.linspace(min(z_Dv)*0.9, max(z_Dv)*1.1, 200)
    DV_plot = np.array([DR2.Dv_over_rd(z_val, pars_fit) for z_val in z_plot_DV])
    axs[2].errorbar(z_Dv, DV_over_rd_exp, yerr=sigma_DV_over_rd, fmt='o', capsize=5, color='darkgreen')
    axs[2].plot(z_plot_DV, DV_plot, 'r-', linewidth=2)
    axs[2].set_xlabel('Redshift $z$')
    axs[2].set_ylabel(r'$D_v / r_d$')
    axs[2].grid(True, alpha=0.3)
    
    param_text = (f'Fit: $\Omega_m={m.values["Omega_m"]:.3f}$, '
                  f'$\Omega_\Lambda={pars_fit["Omega_Lambda"]:.3f}$,\n'
                  f'$w_0={m.values["W_0"]:.2f}$, '
                  f'$w_a={m.values["W_a"]:.2f}$, '
                  f'$\sigma_8={pars_fit["sigma8"]:.2f}$, '
                  f'$H_0r_d={pars_fit["H_0xr_d"]:.2f}$')
    
    fig.text(0.5, 0.98, param_text, 
             ha='center', va='top', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))

    plt.savefig('/home/etudiant15/Documents/STAGE CPPM/Figures/chi_3combined_DESI_BAO+RSD_wCDM.pdf', bbox_inches='tight')
    plt.show()

    merrors_m = m.merrors["Omega_m"]
    lower_m = merrors_m.lower
    upper_m = merrors_m.upper
    inf_m = - lower_m

    merrors_0 = m.merrors["W_0"]
    lower_0 = merrors_0.lower
    upper_0 = merrors_0.upper
    inf_0 = - lower_0

    #merrors_a = m.merrors["W_a"]
    #lower_a = merrors_a.lower
    #upper_a = merrors_a.upper
    #inf_a = - lower_a

    merrors_sigma = m.errors["sigma8"]
    lower_sigma = merrors_sigma
    upper_sigma = merrors_sigma
    inf_sigma = - lower_sigma

    merrors_H = m.merrors["H_0xr_d"]
    lower_H = merrors_H.lower
    upper_H = merrors_H.upper
    inf_H = - lower_H

    print(f'BAO+RSD & '+
        f'${m.values["Omega_m"]:.3f}^{{+{upper_m:.3f}}}_{{{- inf_m:.3f}}}$ &'+
        f' ${m.values["W_0"]:.3f}^{{+{upper_0:.3f}}}_{{{- inf_0:.3f}}}$ & 0 & '+
        f'${m.values["sigma8"]:.3f}^{{+{upper_sigma:.3f}}}_{{{inf_sigma:.3f}}}$ & '+
        f'${m.values["H_0xr_d"]:.3f}^{{+{upper_H:.3f}}}_{{{- inf_H:.3f}}}$ & '+
        f'{m.fval:.2f} & {m.ndof}')

    return m, pars_fit

plot_fit_DR1_DR2_noPV_wCDM()