import matplotlib.pyplot as plt
import numpy as np
import fonctions, DESI_DR1__measurements as DR1, fit_DESI_DR2_BAO_measurements as DR2
import pandas as pd
from iminuit.cost import LeastSquares
from matplotlib.lines import Line2D
import emcee
import corner
import getdist
from getdist import MCSamples, plots
import os

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

# avec PV
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
#----------------------------
para_names_w0wa = ["Omega_m", "W_0", "W_a", "sigma8", "H_0xr_d"]
para_names_wCDM = ["Omega_m", "W_0", "sigma8", "H_0xr_d"]
para_names_BAO_w0wa = ["Omega_m", "W_0", "W_a", "H_0xr_d"]
para_names_BAO_wCDM = ["Omega_m", "W_0", "H_0xr_d"]


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

def chi_carré_fsigma8(pars):
    sum = 0
    for i in range(len(z)):
        theo_val = fonctions.fsigma8_th(z[i], pars)
        if np.isnan(theo_val) or np.isinf(theo_val):
            return np.inf
        
        exp_val = fsigma8_exp[i]
        sigma = (sigma_fsigma8[i])

        sum += ((exp_val - theo_val) ** 2) / (sigma ** 2)
    return sum

def chi_carré_fsigma8_noPV(pars):
    sum = 0
    for i in range(len(z_noPV)):
        theo_val = fonctions.fsigma8_th(z_noPV[i], pars)
        if np.isnan(theo_val) or np.isinf(theo_val):
            return np.inf
        
        exp_val = fsigma8_exp_noPV[i]
        sigma = (sigma_fsigma8_noPV[i])

        sum += ((exp_val - theo_val) ** 2) / (sigma ** 2)
    return sum

#QUE DR2

def chi_carré_BAO(pars):
    return chi_carré_DM_over_DH(pars) + chi_carré_Dv_over_rd(pars)

def chi_carré_BAO_RSD_PV(pars):
    return chi_carré_DM_over_DH(pars) + chi_carré_Dv_over_rd(pars) + chi_carré_fsigma8(pars)

def chi_carré_BAO_RSD(pars):
    return chi_carré_DM_over_DH(pars) + chi_carré_Dv_over_rd(pars) + chi_carré_fsigma8_noPV(pars)


def log_prior_BAO_w0wa(p, limits):
    for i, param in enumerate(para_names_BAO_w0wa):
        low, high = limits[param]
        if p[i] < low or p[i] > high:
            return -np.inf
    return 0.0
    
def log_prior_BAO_wCDM(p, limits):
    for i, param in enumerate(para_names_BAO_wCDM):
        low, high = limits[param]
        if p[i] < low or p[i] > high:
            return -np.inf
    else:
        return 0
 
def log_prior_w0wa(p, limits):
    for i, param in enumerate(para_names_w0wa):
        low, high = limits[param]
        if p[i] < low or p[i] > high:
            return -np.inf
    return 0.0
    
def log_prior_wCDM(p, limits):
    for i, param in enumerate(para_names_wCDM):
        low, high = limits[param]
        if p[i] < low or p[i] > high:
            return -np.inf
    else:
        return 0
 

# prior w0wa
def log_prob_BAO_w0wa(p, limits):
    pars = {
        "Omega_m": p[0],
        #"Omega_Lambda": 1 - p[0],
        "W_0": p[1],
        "W_a": p[2],
        "H_0": 73.2,
        "sigma8": 0.8,
        "H_0xr_d": p[3],
    }
    return -chi_carré_BAO(pars) / 2 + log_prior_BAO_w0wa(p, limits)

def log_prob_BAO_RSD_w0wa(p, limits):
    pars = {
        "Omega_m": p[0],
        "W_0": p[1],
        "W_a": p[2],
        "H_0": 73.2,
        "sigma8": p[3],
        "H_0xr_d": p[4],
    }
    return -chi_carré_BAO_RSD(pars) / 2 + log_prior_w0wa(p, limits)

def log_prob_BAO_RSD_PV_w0wa(p, limits):
    pars = {
        "Omega_m": p[0],
        "W_0": p[1],
        "W_a": p[2],
        "H_0": 73.2,
        "sigma8": p[3],
        "H_0xr_d": p[4],
    }
    return -chi_carré_BAO_RSD_PV(pars) / 2 + log_prior_w0wa(p, limits)

# prior wCDM
def log_prob_BAO_wCDM(p, limits):
    pars = {
        "Omega_m": p[0],
        #"Omega_Lambda": 1 - p[0],
        "W_0": p[1],
        "W_a": 0,
        "H_0": 73.2,
        "sigma8": 0.8,
        "H_0xr_d": p[2],
    }
    return -chi_carré_BAO(pars) / 2 + log_prior_BAO_wCDM(p, limits)

def log_prob_BAO_RSD_wCDM(p, limits):
    pars = {
        "Omega_m": p[0],
        "W_0": p[1],
        "W_a": 0,
        "H_0": 73.2,
        "sigma8": p[2],
        "H_0xr_d": p[3],
    }
    return -chi_carré_BAO_RSD(pars) / 2 + log_prior_wCDM(p, limits)

def log_prob_BAO_RSD_PV_wCDM(p, limits):
    pars = {
        "Omega_m": p[0],
        #"Omega_Lambda": 1 - p[0],
        "W_0": p[1],
        "W_a": 0,
        "H_0": 73.2,
        "sigma8": p[2],
        "H_0xr_d": p[3],
    }
    return -chi_carré_BAO_RSD_PV(pars) / 2 + log_prior_wCDM(p, limits)

# BAO
def mcmc_BAO_w0wa():
    param_names = para_names_BAO_w0wa
    ndim = len(param_names)
    nwalkers = 10
    limits = {}
    limits["Omega_m"] = (0.1, 1.0)
    limits["W_0"] = (-3.0, 1.0)
    limits["W_a"] = (-3.0, 2.0)
    #limits["H_0"] = (73.2, 73.2) #on fixe H_0 à 73.2
    limits["H_0xr_d"] = (5000, 20000)
    pmin = np.array([])
    pmax = np.array([])
    for param in param_names:
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
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_BAO_w0wa, args=[limits])
    #log_prob(p0[0], limits)
    #print(log_prob(p0[0], limits))

    state = sampler.run_mcmc(p0, 1000)
    sampler.run_mcmc(state, 100, progress=True)
    samples = sampler.get_chain(flat=True)
    np.save('mes_chaines_BAO_w0wa.npy', samples)
mcmc_BAO_w0wa()
"""def plot_mcmc_BAO_w0wa():
    samples = np.load('mes_chaines_BAO_w0wa.npy')
    n_cols = samples.shape[1]
    #fig, axes = plt.subplots(5, figsize=(10, 10), sharex=True)
    #print(f"Shape de samples: {samples.shape}")
    #print(f"Nombre d'échantillons: {len(samples)}")
    #print("n_cols:", n_cols)
    #print("ndim:", ndim)
    #for i in range(ndim):
    #    ax = axes[i]
    #    ax.plot(samples[:, i], "k", alpha=0.3)
    #    ax.set_xlim(0, len(samples))
    #    ax.set_ylabel(para_names[i])
    #    ax.yaxis.set_label_coords(-0.1, 0.5)
    #axes[-1].set_xlabel("step number")
    fig = corner.corner(samples[:, :5], labels=para_names_w0wa, )
    plt.savefig(
        "/home/etudiant15/Documents/STAGE CPPM/Figures/MCMC_BAO_w0waCDM.pdf",
        bbox_inches="tight",
    )
    plt.tight_layout()
    plt.show()"""

def plot_mcmc_BAO_w0wa():
    samples = np.load('mes_chaines_BAO_w0wa.npy')

    if len(samples) > 50000:
        samples = samples[::4]  # 1 point sur 4
    elif len(samples) > 100000:
        samples = samples[::2]

    param_names = para_names_BAO_w0wa
    labels = para_names_BAO_w0wa
    samples_getdist = MCSamples(
        samples=samples,
        names=param_names,
        labels=labels,
        sampler='mcmc',
        settings={'fine_bins': 512,  # Augmenter le nombre de bins
        'fine_bins_2D': 64,  # Augmenter pour les plots 2D
        'smooth_scale_1D': 0.2,  # smoothing manuel
        'smooth_scale_2D': 0.3,  # manuel
        'num_bins': 20})
    
    g = plots.get_subplot_plotter()
    g.settings.axes_fontsize = 10
    g.settings.title_limit_fontsize = 10
    #g.settings.legend_fontsize = 12

    g.triangle_plot([samples_getdist], filled=True, contour_colors=['blue'])
    plt.savefig(
        "/home/etudiant15/Documents/STAGE CPPM/Figures/MCMC_BAO_w0waCDM.pdf",
        bbox_inches="tight",)
    plt.show()


  
def mcmc_BAO_wCDM():
    param_names = para_names_BAO_wCDM
    ndim = len(param_names)
    nwalkers = 10
    limits = {}
    limits["Omega_m"] = (0.1, 1.0)
    limits["W_0"] = (-3.0, 1.0)
    limits["H_0xr_d"] = (5000, 20000)
    pmin = np.array([])
    pmax = np.array([])
    for param in param_names:
        pmin = np.append(pmin, limits[param][0])
        pmax = np.append(pmax, limits[param][1])
    p0 = pmin + np.random.rand(nwalkers, ndim) * (pmax - pmin)  # nwalkers entre 0 et 1
    for j in range(ndim):
        max = p0[:,j].max()
        min = p0[:,j].min()
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_BAO_wCDM, args=[limits])
    #log_prob(p0[0], limits)
    state = sampler.run_mcmc(p0, 1000)
    sampler.run_mcmc(state, 100, progress=True)
    samples = sampler.get_chain(flat=True)
    np.save('mes_chaines_BAO_wCDM.npy', samples)

def plot_mcmc_BAO_wCDM():
    samples = np.load('mes_chaines_BAO_wCDM.npy')

    if len(samples) > 50000:
        samples = samples[::4]  # 1 point sur 10
    elif len(samples) > 100000:
        samples = samples[::2]

    param_names = para_names_BAO_wCDM
    labels = para_names_BAO_wCDM
    samples_getdist = MCSamples(
        samples=samples,
        names=param_names,
        labels=labels,
        sampler='mcmc',
        settings={'fine_bins': 512,  # Augmenter le nombre de bins
        'fine_bins_2D': 64,  # Augmenter pour les plots 2D
        'smooth_scale_1D': 0.2,  # smoothing manuel
        'smooth_scale_2D': 0.3,  # manuel
        'num_bins': 20})
    
    g = plots.get_subplot_plotter()
    g.settings.axes_fontsize = 10
    g.settings.title_limit_fontsize = 10

    g.triangle_plot([samples_getdist], filled=True, contour_colors=['blue'])
    
    plt.savefig(
        "/home/etudiant15/Documents/STAGE CPPM/Figures/MCMC_BAO_wCDM.pdf", bbox_inches="tight",)
    plt.show()



# BAO + RSD
def mcmc_BAO_RSD_w0wa():
    param_names = para_names_w0wa
    ndim = len(param_names)
    nwalkers = 100
    limits = {}
    limits["Omega_m"] = (0.1, 1.0)
    limits["W_0"] = (-3.0, 1.0)
    limits["W_a"] = (-3.0, 2.0)
    limits["sigma8"] = (0.6, 1.0)
    limits["H_0xr_d"] = (5000, 20000)
    pmin = np.array([])
    pmax = np.array([])
    for param in param_names:
        pmin = np.append(pmin, limits[param][0])
        pmax = np.append(pmax, limits[param][1])
    p0 = pmin + np.random.rand(nwalkers, ndim) * (pmax - pmin)  # nwalkers entre 0 et 1

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_BAO_RSD_w0wa, args=[limits])
    #log_prob(p0[0], limits)
    state = sampler.run_mcmc(p0, 500)
    sampler.run_mcmc(state, 5000, progress=True)
    samples = sampler.get_chain(flat=True)
    np.save('mes_chaines_BAO_RSD_w0wa.npy', samples)

def plot_mcmc_BAO_RSD_w0wa():
    samples = np.load('mes_chaines_BAO_RSD_w0wa.npy')

    if len(samples) > 50000:
        samples = samples[::4]  # 1 point sur 10
    elif len(samples) > 100000:
        samples = samples[::2]

    param_names = para_names_w0wa
    labels = para_names_w0wa
    samples_getdist = MCSamples(
        samples=samples,
        names=param_names,
        labels=labels,
        sampler='mcmc',
        settings={'fine_bins': 512,  # Augmenter le nombre de bins
        'fine_bins_2D': 64,  # Augmenter pour les plots 2D
        'smooth_scale_1D': 0.2,  # smoothing manuel
        'smooth_scale_2D': 0.3,  # manuel
        'num_bins': 20})  
    
    g = plots.get_subplot_plotter()
    g.settings.axes_fontsize = 10
    g.settings.title_limit_fontsize = 10
    #g.settings.legend_fontsize = 12

    g.triangle_plot([samples_getdist], filled=True, contour_colors=['green'])
    plt.savefig(
        "/home/etudiant15/Documents/STAGE CPPM/Figures/MCMC_BAO_RSD_w0waCDM.pdf", bbox_inches="tight",)
    plt.show()



def mcmc_BAO_RSD_wCDM():
    param_names = para_names_wCDM
    ndim = len(param_names)
    nwalkers = 10
    limits = {}
    limits["Omega_m"] = (0.1, 1.0)
    limits["W_0"] = (-3.0, 1.0)
    limits["sigma8"] = (0.6, 1.0)
    limits["H_0xr_d"] = (5000, 20000)
    pmin = np.array([])
    pmax = np.array([])
    for param in param_names:
        pmin = np.append(pmin, limits[param][0])
        pmax = np.append(pmax, limits[param][1])
    p0 = pmin + np.random.rand(nwalkers, ndim) * (pmax - pmin)  # nwalkers entre 0 et 1
    for j in range(ndim):
        max = p0[:,j].max()
        min = p0[:,j].min()
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_BAO_RSD_wCDM, args=[limits])
    #log_prob(p0[0], limits)
    state = sampler.run_mcmc(p0, 1000)
    sampler.run_mcmc(state, 100, progress=True)
    samples = sampler.get_chain(flat=True)
    np.save('mes_chaines_BAO_RSD_wCDM.npy', samples)

def plot_mcmc_BAO_RSD_wCDM():
    samples = np.load('mes_chaines_BAO_RSD_wCDM.npy')

    if len(samples) > 50000:
        samples = samples[::4]  # 1 point sur 4
    elif len(samples) > 100000:
        samples = samples[::2]


    param_names = para_names_wCDM
    labels = para_names_wCDM
    samples_getdist = MCSamples(
        samples=samples,
        names=param_names,
        labels=labels,
        sampler='mcmc',
        settings={'fine_bins': 512, 'fine_bins_2D': 64,  # Augmenter pour les plots 2D
        'smooth_scale_1D': 0.2,  # smoothing manuel
        'smooth_scale_2D': 0.3,  # manuel
        'num_bins': 20})
    
    g = plots.get_subplot_plotter()
    g.settings.axes_fontsize = 10
    g.settings.title_limit_fontsize = 10
    #g.settings.legend_fontsize = 12

    g.triangle_plot([samples_getdist], filled=True, contour_colors=['green'])
    plt.savefig(
        "/home/etudiant15/Documents/STAGE CPPM/Figures/MCMC_BAO_RSD_wCDM.pdf", bbox_inches="tight",)
    plt.show()


# BAO + RSD + PV
def mcmc_BAO_RSD_PV_w0wa():
    param_names = para_names_w0wa
    ndim = len(param_names)
    nwalkers = 10
    limits = {}
    limits["Omega_m"] = (0.1, 1.0)
    limits["W_0"] = (-3.0, 1.0)
    limits["W_a"] = (-3.0, 2.0)
    limits["sigma8"] = (0.6, 1.0)
    limits["H_0xr_d"] = (5000, 20000)
    pmin = np.array([])
    pmax = np.array([])
    for param in param_names:
        pmin = np.append(pmin, limits[param][0])
        pmax = np.append(pmax, limits[param][1])
    p0 = pmin + np.random.rand(nwalkers, ndim) * (pmax - pmin)  # nwalkers entre 0 et 1
    for j in range(ndim):
        max = p0[:,j].max()
        min = p0[:,j].min()
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_BAO_RSD_PV_w0wa, args=[limits])
    #log_prob(p0[0], limits)
    state = sampler.run_mcmc(p0, 1000)
    sampler.run_mcmc(state, 100, progress=True)
    samples = sampler.get_chain(flat=True)
    np.save('mes_chaines_BAO_RSD_PV_w0wa.npy', samples)

def plot_mcmc_BAO_RSD_PV_w0wa():
    samples = np.load('mes_chaines_BAO_RSD_PV_w0wa.npy')

    if len(samples) > 50000:
        samples = samples[::4]  # 1 point sur 4
    elif len(samples) > 100000:
        samples = samples[::2]

    param_names = para_names_w0wa
    labels = para_names_w0wa
    samples_getdist = MCSamples(
        samples=samples,
        names=param_names,
        labels=labels,
        sampler='mcmc',
        settings={'fine_bins': 512, 
                  'fine_bins_2D': 64,
                  'smooth_scale_1D': 0.2,
                  'smooth_scale_2D': 0.3,
                  'num_bins': 20})
    
    g = plots.get_subplot_plotter()
    g.settings.axes_fontsize = 10
    g.settings.title_limit_fontsize = 10
    #g.settings.legend_fontsize = 12

    g.triangle_plot([samples_getdist], filled=True, contour_colors=['red'])
    plt.savefig("/home/etudiant15/Documents/STAGE CPPM/Figures/MCMC_BAO_RSD_PV_w0waCDM.pdf", bbox_inches="tight",)
    plt.show()


def mcmc_BAO_RSD_PV_wCDM():
    param_names = para_names_wCDM
    ndim = len(param_names)
    nwalkers = 10
    limits = {}
    limits["Omega_m"] = (0.1, 1.0)
    limits["W_0"] = (-3.0, 1.0)
    limits["sigma8"] = (0.6, 1.0)
    limits["H_0xr_d"] = (5000, 20000)
    pmin = np.array([])
    pmax = np.array([])
    for param in param_names:
        pmin = np.append(pmin, limits[param][0])
        pmax = np.append(pmax, limits[param][1])
    p0 = pmin + np.random.rand(nwalkers, ndim) * (pmax - pmin)  # nwalkers entre 0 et 1
    """for j in range(ndim_wCDM):
        max = p0[:,j].max()
        min = p0[:,j].min()"""
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_BAO_RSD_PV_wCDM, args=[limits])
    #log_prob(p0[0], limits)
    state = sampler.run_mcmc(p0, 1000)
    sampler.run_mcmc(state, 100, progress=True)
    samples = sampler.get_chain(flat=True)
    np.save('mes_chaines_BAO_RSD_PV_wCDM.npy', samples)

def plot_mcmc_BAO_RSD_PV_wCDM():
    samples = np.load('mes_chaines_BAO_RSD_PV_wCDM.npy')

    if len(samples) > 50000:
        samples = samples[::4]  # 1 point sur 4
    elif len(samples) > 100000:
        samples = samples[::2]

    param_names = para_names_wCDM
    labels = para_names_wCDM
    samples_getdist = MCSamples(
        samples=samples,
        names=param_names,
        labels=labels,
        sampler='mcmc',
        settings={'fine_bins': 512, 'fine_bins_2D': 64,  # Augmenter pour les plots 2D
        'smooth_scale_1D': 0.2,  # smoothing manuel
        'smooth_scale_2D': 0.3,  # manuel
        'num_bins': 20})
    
    g = plots.get_subplot_plotter()
    g.settings.axes_fontsize = 10
    g.settings.title_limit_fontsize = 10
    #g.settings.legend_fontsize = 12

    g.triangle_plot([samples_getdist], filled=True, contour_colors=['red'])
    plt.savefig(
        "/home/etudiant15/Documents/STAGE CPPM/Figures/MCMC_BAO_RSD_PV_wCDM.pdf", bbox_inches="tight",)
    plt.show()

# p defini avc paramètres dans l'ordre

def plot_mcmc_w0wa():
    samples_BAO = np.load('mes_chaines_BAO_w0wa.npy')
    samples_BAO_RSD = np.load('mes_chaines_BAO_RSD_w0wa.npy')
    samples_BAO_RSD_PV = np.load('mes_chaines_BAO_RSD_PV_w0wa.npy')

    samples_list = [samples_BAO, samples_BAO_RSD, samples_BAO_RSD_PV]

    for i in range(len(samples_list)):
        if len(samples_list[i]) > 100000:
            samples_list[i] = samples_list[i][::4]
        elif len(samples_list[i]) > 50000:
            samples_list[i] = samples_list[i][::2]
    

    mcsamples_list = []
    colors = ['blue', 'green', 'red']  # Couleurs différentes pour chaque dataset
    legends = ['BAO only', 'BAO+RSD', 'BAO+RSD+PV']

    labels_latex = [r'$w_0$', r'$w_a$', r'$\Omega_m$', r'$h$']
    labels_latex_extended = [r'$w_0$', r'$w_a$', r'$\Omega_m$', r'$h$', r'$\sigma_8$']

    for i, samples in enumerate(samples_list):
        if i == 0 : 
            param = para_names_BAO_w0wa
            labels = labels_latex[:len(param)]
        else : 
            param = para_names_w0wa
            labels = labels_latex_extended[:len(param)]

        samples_mcsamples = MCSamples(
            samples=samples,
            names=param,
            labels=param,
            label=legends[i],
            sampler='mcmc',
            settings={'fine_bins': 512,
                    'fine_bins_2D': 64,
                    'smooth_scale_1D': 0.2,
                    'smooth_scale_2D': 0.3,
                    'num_bins': 20})
        mcsamples_list.append(samples_mcsamples) 
    
    g = plots.get_subplot_plotter()
    g.settings.axes_fontsize = 10
    g.settings.title_limit_fontsize = 10
    g.settings.legend_fontsize = 12
    g.settings.legend_loc = 'upper right'

    g.triangle_plot(mcsamples_list, filled=True, contour_colors=colors, legend_labels=legends, legend_loc='upper right')
    plt.savefig(
        "/home/etudiant15/Documents/STAGE CPPM/Figures/MCMC_w0waCDM.pdf", bbox_inches="tight",)
    plt.show()

def plot_mcmc_wCDM():
    samples_BAO = np.load('mes_chaines_BAO_wCDM.npy')
    samples_BAO_RSD = np.load('mes_chaines_BAO_RSD_wCDM.npy')
    samples_BAO_RSD_PV = np.load('mes_chaines_BAO_RSD_PV_wCDM.npy')

    samples_list = [samples_BAO, samples_BAO_RSD, samples_BAO_RSD_PV]

    for i in range(len(samples_list)):
        if len(samples_list[i]) > 100000:
            samples_list[i] = samples_list[i][::4]
        elif len(samples_list[i]) > 50000:
            samples_list[i] = samples_list[i][::2]
    

    mcsamples_list = []
    colors = ['blue', 'green', 'red']  # Couleurs différentes pour chaque dataset
    legends = ['BAO only', 'BAO+RSD', 'BAO+RSD+PV']

    labels_latex = [r'$w_0$', r'$\Omega_m$', r'$h$']
    labels_latex_extended = [r'$w_0$', r'$\Omega_m$', r'$h$', r'$\sigma_8$']

    for i, samples in enumerate(samples_list):
        if i == 0 : 
            param = para_names_BAO_wCDM
            labels = labels_latex[:len(param)]
        else : 
            param = para_names_wCDM
            labels = labels_latex_extended[:len(param)]

        samples_mcsamples = MCSamples(
            samples=samples,
            names=param,
            labels=param,
            label=legends[i],
            sampler='mcmc',
            settings={'fine_bins': 512,
                    'fine_bins_2D': 64,
                    'smooth_scale_1D': 0.2,
                    'smooth_scale_2D': 0.3,
                    'num_bins': 20})
        mcsamples_list.append(samples_mcsamples) 
    
    g = plots.get_subplot_plotter()
    g.settings.axes_fontsize = 10
    g.settings.title_limit_fontsize = 10
    g.settings.legend_fontsize = 12
    g.settings.legend_loc = 'upper right'

    g.triangle_plot(mcsamples_list, filled=True, contour_colors=colors, legend_labels=legends, legend_loc='upper right')
    plt.savefig(
        "/home/etudiant15/Documents/STAGE CPPM/Figures/MCMC_wCDM.pdf", bbox_inches="tight",)
    plt.show()

a = np.load('mes_chaines_BAO_wCDM.npy')
a_ = np.load('mes_chaines_BAO_w0wa.npy')
b = np.load('mes_chaines_BAO_RSD_wCDM.npy')
b_ = np.load('mes_chaines_BAO_RSD_w0wa.npy')
c = np.load('mes_chaines_BAO_RSD_PV_wCDM.npy')
c_ = np.load('mes_chaines_BAO_RSD_PV_w0wa.npy')
print("longueur BAO wCDM:", len(a), "longueur BAO w0waCDM:",len(a_), len(b), len(b_), len(c), len(c_))
