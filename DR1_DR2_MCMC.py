import matplotlib.pyplot as plt
import numpy as np
import fonctions, DESI_DR1__measurements as DR1, fit_DESI_DR2_BAO_measurements as DR2
import pandas as pd
from iminuit.cost import LeastSquares
from matplotlib.lines import Line2D
import emcee

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

def chi_carré_Dv_over_rd(pars):
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


def chi_carré(pars):
    return chi_carré_DM_over_DH(pars) + chi_carré_Dv_over_rd(pars)

def log_prior(p, limits):
    for i, param in enumerate(para_names):
        if p[i] < limits[param][0] or p[i] > limits[param][1]:
            return -np.inf
    else : 
        return 0
 
    return 0
def log_prob(p, limits):
    pars = {
        "Omega_m": p[0],
        #"Omega_Lambda": 1 - p[0],
        "W_0": p[1],
        "W_a": p[2],
        "H_0": 73.2,
        "sigma8": p[3],
        "H_0xr_d": p[4],
    }
    return -chi_carré(pars) / 2 + log_prior(p, limits)


def mcmc_BAO_RSD_PV_w0wa():
    nwalkers = 10
    ndim = len(para_names)

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
        print("Le max du paramètre", para_names[j], "est :", max)
        print("Le min du paramètre", para_names[j], "est :", min)

    #print("p0", p0.shape)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[limits])
    #log_prob(p0[0], limits)
    #print(log_prob(p0[0], limits))
    state = sampler.run_mcmc(p0, 100)
    sampler.run_mcmc(state, 10000)
    """sampler.reset()
    sampler.run_mcmc(state, 10000)
    samples = sampler.get_chain(flat=True)
    plt.hist(samples[:, 0], 100, color="k", histtype="step")
    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$p(\theta_1)$")
    plt.gca().set_yticks([]);"""

mcmc_BAO_RSD_PV_w0wa()
# p defini avc paramètres dans l'ordre
