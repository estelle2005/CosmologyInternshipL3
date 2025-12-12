import matplotlib.pyplot as plt
import numpy as np
import fonctions, DESI_DR1__measurements as DR1, fit_DESI_DR2_BAO_measurements as DR2
import pandas as pd
from iminuit.cost import LeastSquares
from matplotlib.lines import Line2D
import emcee
import corner

r_d = 147.05  # Mpc today
c = 3 * 10**5  # en km

tableau_DR2 = pd.read_csv("DESI_DR2_BAO_measurements.csv")
tableau_DR2 = tableau_DR2.sort_values("z_eff").reset_index(drop=True)

z_Dv = tableau_DR2["z_eff"].to_numpy()
DV_over_rd_exp = tableau_DR2["DV_over_rd"].to_numpy()
sigma_DV_over_rd = tableau_DR2["DV_over_rd_err"].to_numpy()

z_DM = tableau_DR2["z_eff"].to_numpy()[1:]
DM_over_DH_exp = tableau_DR2["DM_over_DH"].to_numpy()[1:]
sigma_DM_over_DH = tableau_DR2["DM_over_DH_err"].to_numpy()[1:]


tableau_DR1 = pd.read_csv("fs8_DESI_DR1_RSD+PV.csv")
tableau_DR1 = tableau_DR1.sort_values("z_eff").reset_index(drop=True)

z = tableau_DR1["z_eff"].to_numpy()
fsigma8_exp = tableau_DR1["fsigma8"].to_numpy()
sigma_fsigma8 = tableau_DR1["fsigma8_err"].to_numpy()

# sans PV
tableau_DR1_noPV = pd.read_csv("fs8_DESI_DR1_RSDnoPV.csv")
tableau_DR1_noPV = tableau_DR1_noPV.sort_values("z_eff").reset_index(drop=True)

z_noPV = tableau_DR1_noPV["z_eff"].to_numpy()
fsigma8_exp_noPV = tableau_DR1_noPV["fsigma8"].to_numpy()
sigma_fsigma8_noPV = tableau_DR1_noPV["fsigma8_err"].to_numpy()

para_names = ["Omega_m", "W_0", "W_a", "sigma8", "H_0xr_d"]
ndim = len(para_names)
"""def chi_carré_Dv_over_rd(pars):
    sum = 0
    for i in range(len(z_Dv)):
        sum += ((DV_over_rd_exp[i] - fonctions.Dv_over_rd(z_Dv[i], pars)) ** 2) / (
            (sigma_DV_over_rd[i]) ** 2
        )
    return sum

def chi_carré_DM_over_DH(pars):
    sum = 0
    for i in range(len(z_DM)):
        sum += ((DM_over_DH_exp[i] - fonctions.DM_over_DH(z_DM[i], pars)) ** 2) / (
            (sigma_DM_over_DH[i]) ** 2
        )
    return sum
"""

def chi_carré_Dv_over_rd(pars):
    sum = 0
    for i in range(len(z_Dv)):
        theo_val = fonctions.Dv_over_rd(z_Dv[i], pars)
        if np.isnan(theo_val) or np.isinf(theo_val):
            return np.inf
        
        exp_val = DV_over_rd_exp[i]
        sigma = sigma_DV_over_rd[i]

        if np.isnan(exp_val) or sigma <= 0:
            return np.inf

        sum += ((exp_val - theo_val) ** 2) / (sigma ** 2)
    return sum

def chi_carré_DM_over_DH(pars):
    sum = 0
    for i in range(len(z_DM)):
        theo_val = fonctions.DM_over_DH(z_DM[i], pars)
        if np.isnan(theo_val) or np.isinf(theo_val):
            return np.inf
        
        exp_val = DM_over_DH_exp[i]
        sigma = (sigma_DM_over_DH[i])

        sum += ((exp_val - theo_val) ** 2) / (sigma ** 2)
    return sum


#W_0_list = [-1, -0.6, -0.2]
#W_a_list = [0, -1.2, -2.4]
#Omega_m_list = [0.1, 0.3, 0.9]
#for i in range(len(Omega_m_list)):
#    pars = {"Omega_m": Omega_m_list[i],
#            "Omega_Lambda": 1 - Omega_m_list[i],
#            "W_0": W_0_list[i],
#            "W_a": W_a_list[i],
#            "H_0": 73.2, "H_0xr_d":10764.06,}
#    chi_carré_DM_over_DH(pars)
#    chi_carré_Dv_over_rd(pars)
#
#QUE DR2

def chi_carré_BAO(pars):
    return chi_carré_DM_over_DH(pars) + chi_carré_Dv_over_rd(pars)

def log_prior_BAO(p, limits):
    for i, param in enumerate(para_names):
        low, high = limits[param]
        if p[i] < low or p[i] > high:
            return -np.inf
    else:
        return 0
 
def log_prob_BAO(p, limits):
    pars = {
        "Omega_m": p[0],
        #"Omega_Lambda": 1 - p[0],
        "W_0": p[1],
        "W_a": p[2],
        "H_0": 73.2,
        "sigma8": p[3],
        "H_0xr_d": p[4],
    }
    return -chi_carré_BAO(pars) / 2 + log_prior_BAO(p, limits)


def mcmc_BAO_w0wa():
    nwalkers = 10
    limits = {}
    limits["Omega_m"] = (0.1, 1.0)
    limits["W_0"] = (-3.0, 1.0)
    limits["W_a"] = (-3.0, 2.0)
    #limits["H_0"] = (73.2, 73.2) #on fixe H_0 à 73.2
    limits["sigma8"] = (0.6, 1.0)
    limits["H_0xr_d"] = (5000, 20000)
    pmin = np.array([])
    pmax = np.array([])
    for param in para_names:
        # print(param)
        # print("limit paramètre 0:", limits[param][0])
        # print("limit paramètre 1:", limits[param][1])
        pmin = np.append(pmin, limits[param][0])
        pmax = np.append(pmax, limits[param][1])
        # print("pmin:", pmin)
        # print("pmax", pmax)
    p0 = pmin + np.random.rand(nwalkers, ndim) * (pmax - pmin)  # nwalkers entre 0 et 1
    for j in range(ndim):
        max = p0[:,j].max()
        min = p0[:,j].min()
        #print("Le max du paramètre", para_names[j], "est :", max)
        #print("Le min du paramètre", para_names[j], "est :", min)

    #print("p0", p0.shape)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_BAO, args=[limits])
    #log_prob(p0[0], limits)
    #print(log_prob(p0[0], limits))
    state = sampler.run_mcmc(p0, 10000)
    sampler.run_mcmc(state, 100, progress=True)
    samples = sampler.get_chain(flat=True)
    np.save('mes_chaines_BAO_w0wa.npy', samples)
    """sampler.reset()
    sampler.run_mcmc(state, 10000)"""

def plot_mcmc_BAO_w0wa():
    samples = np.load('mes_chaines_BAO_w0wa.npy')
    n_cols = samples.shape[1]
    #fig, axes = plt.subplots(5, figsize=(10, 10), sharex=True)
    #print(f"Shape de samples: {samples.shape}")
    #print(f"Nombre d'échantillons: {len(samples)}")
    #print("n_cols:", n_cols)
    #print("ndim:", ndim)
    """for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(para_names[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")"""
    fig = corner.corner(samples[:, :5], labels=para_names, )
    plt.savefig(
        "/home/etudiant15/Documents/STAGE CPPM/Figures/MCMC_BAO_w0waCDM.pdf",
        bbox_inches="tight",
    )
    plt.tight_layout()
    plt.show()

""" fig = corner.corner(
        samples[:, :5],  # Prend seulement les 5 premières colonnes
        labels=para_names,
        show_titles=True,
        plot_datapoints=False
    )"""
  

mcmc_BAO_w0wa()
# p defini avc paramètres dans l'ordre
#faire apres avec DR1 avec avec et sans PV et w0waCDM et wCDM
