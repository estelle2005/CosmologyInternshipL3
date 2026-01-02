import matplotlib.pyplot as plt
import numpy as np
import logging
from scipy.integrate import odeint
from scipy.integrate import quad


c = 3 * 10**5  # en km

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# $W(a) = W_0 + (1-a)W_a $

# -- Definition of \"time\" = ln(a)
# a = 10.**np.linspace(-2, 0, 10000)  #de 10**-2 à 10**0
# ln_a = np.log(a)
# z = 1/a - 1

# pars = {'Omega_Lambda': Omega_Lambda, 'W_0': W_0_list[i], 'W_a': W_a_list[i]}


"""
def H(a, pars):
    Omega_m = pars["Omega_m"]
    W_0 = pars["W_0"]
    W_a = pars["W_a"]
    H_0 = pars["H_0"]
    Omega_Lambda = 1 - Omega_m
    return H_0 * np.sqrt(Omega_m * a**-3 + Omega_Lambda * a ** (-3 * (1 + W_0 + W_a)) * np.exp(-3 * W_a * (1 - a)))
"""

def H(a, pars):
    Omega_m = pars["Omega_m"]
    W_0 = pars["W_0"]
    W_a = pars["W_a"]
    H_0 = pars["H_0"]
    Omega_Lambda = 1 - Omega_m
    if a <= 0:
        return np.inf
    try:
        inside_sqrt = Omega_m * a**-3 + Omega_Lambda * a ** (-3 * (1 + W_0 + W_a)) * np.exp(-3 * W_a * (1 - a))
        if inside_sqrt <= 0:
            return 1e-10
        return H_0 * np.sqrt(inside_sqrt)
    except (OverflowError, FloatingPointError):
        return np.inf


def H_prime(a, pars):
    Omega_m = pars["Omega_m"]
    W_0 = pars["W_0"]
    W_a = pars["W_a"]
    H_0 = pars["H_0"]
    Omega_Lambda = 1 - Omega_m
    u_prime = Omega_m * a ** (-3) + (1 + W_0 + W_a - W_a * a) * Omega_Lambda * a ** (
        -3 * (1 + W_0 + W_a)
    ) * np.exp(-3 * W_a * (1 - a))
    H_prime = -(H_0**2) * 3 / 2 * u_prime / H(a, pars)
    return H_prime


def Omega_m_a(a, pars):
    Omega_m = pars["Omega_m"]
    W_0 = pars["W_0"]
    W_a = pars["W_a"]
    Omega_Lambda = 1 - Omega_m
    return Omega_m / (
        Omega_m + Omega_Lambda * a ** (-3 * (W_0 + W_a)) * np.exp(-3 * W_a * (1 - a))
    )


def df_over_dlna(f, ln_a, pars):
    # f' = df/dlna
    a = np.exp(ln_a)
    # f' = - f² - \left( 2 + \frac{H\prime}{H} \right) f + \frac{3}{2}\Omega_m(a, w_a, w_0)
    deriv = -(f**2) - f * (2 + H_prime(a, pars) / H(a, pars)) + 1.5 * Omega_m_a(a, pars)
    return deriv


def growth_rate_f(z, pars):
    if hasattr(z, "__len__") == True:
        f = np.array([growth_rate_f(z_i, pars) for z_i in z])
    else:
        a_z = 1 / (1 + z)
        a = 10.0 ** np.linspace(
            -3, np.log10(a_z), 1000
        )  # de 10**-2 à a_z qui dépend de z
        ln_a = np.log(a)
        f0 = 1  # condition initiale
        f = odeint(df_over_dlna, f0, ln_a, args=(pars,))[-1, 0]
    return f


"""

def growth_rate_f(z, pars, a_array=None):
    
    #Calcule le taux de croissance f(z) pour un z donné ou un tableau de z.
    
    #Si a_array est fourni, on calcule f sur tout le domaine a_array
    #et on interpole pour obtenir f(z).
    
    if np.isscalar(z) or (hasattr(z, "__len__") and len(z) == 1):
        if np.isscalar(z):
            z_val = z
        else:
            z_val = z[0]
    
        a_z = 1 / (1+z_val)
        if a_z > 0.01:
            a = 10.**np.linspace(-2, np.log10(a_z), 1000)  #de 10**-2 à a_z
        else: 
            a = np.array([0.01, a_z])

        ln_a = np.log(a)
        f0 = 1 #condition initiale
        f_solution = odeint(df_over_dlna, f0, ln_a, args=(pars,))
        return f_solution[-1, 0]
    else:
        # Cas d'un tableau de z - on calcule f pour le plus grand z
        # et on interpole pour les autres
        z_max = np.max(z)
        a_max = 1/(1+z_max)
        if a_max > 0.01:
            a_domain = 10.**np.linspace(-2, np.log10(a_max), 1000)  #de 10**-2 à a_z
        else: 
            a_domain = np.linspace(0.01, a_max, 100)
        
        ln_a_domain = np.log(a_domain)
        f0 = 1
        f_solution = odeint(df_over_dlna, f0, ln_a_domain, args=(pars,)) 
        f_solution = f_solution.flatten()

        a_values = 1/(1+z)

        f_interp = np.interp(a_values, a_domain, f_solution)

        return f_interp
"""

"""def growth_factor_D(z, pars):
    if hasattr(z, "__len__") == True : 
        D = np.array([growth_factor_D(z_i, pars) for z_i in z])
    else:
        a_z = 1 / (1+z)
        a = 10.**np.linspace(-2, np.log10(a_z), 1000)  #de 10**-2 à a qui dépend de z
        ln_a = np.log(a)
        D_init = 0.01   #a_init = 0.01 - comme si on mettait 'A_s', cad on normalise
        delta_lna = ln_a[1] - ln_a[0]
        term = growth_rate_f(z, pars) * delta_lna
        int_dlnD = np.cumsum(term)
        ln_D = int_dlnD + np.log(D_init)
        D = np.exp(ln_D)
    return D"""


def growth_factor_D_calcul(z, pars):
    if hasattr(z, "__len__") == True:
        D = np.array([growth_factor_D_calcul(z_i, pars) for z_i in z])
    else:
        a_z = 1 / (1 + z)
        a = 10.0 ** np.linspace(
            -3, np.log10(a_z), 1000
        )  # de 10**-2 à a qui dépend de z
        ln_a = np.log(a)

        # On a besoin de f sur TOUT le domaine ln_a, pas juste f(z)
        # Donc on doit résoudre l'équation différentielle pour obtenir f(ln_a)
        f0 = 1
        f_solution = odeint(df_over_dlna, f0, ln_a, args=(pars,)).flatten()

        D_init = 0.01
        delta_lna = ln_a[1] - ln_a[0]
        int_dlnD = np.cumsum(f_solution) * delta_lna
        ln_D = int_dlnD + np.log(D_init)
        D = np.exp(ln_D)[-1]  # On veut D au dernier point (a=a_z)
    return D


def growth_factor_D(z, pars):
    return growth_factor_D_calcul(z, pars) / growth_factor_D_calcul(0, pars)


def fsigma8_th(z_val, pars):
    f_solution = growth_rate_f(z_val, pars)
    f_values = f_solution
    return f_values * growth_factor_D(z_val, pars) * pars["sigma8"]


def plot_D():  # D pour différentes valeurs de w_0 et w_a à Omega_Lambda fixé, en fonction de z
    a = 10.0 ** np.linspace(-2, 0, 1000)  # de 10**-2 à 10**0
    z = 1 / a - 1
    plt.figure()  # à vérifier
    W_0_list = [-1, -0.8, -0.6, -0.4, -0.2]
    W_a_list = [0, -0.6, -1.2, -1.8, -2.4]
    # Omega_m = 0.3
    Omega_m = 0.3
    for i in range(len(W_a_list)):
        # logging.info(f"boucle D, {i}")
        pars = {"Omega_m": Omega_m, "W_0": W_0_list[i], "W_a": W_a_list[i], "H_0": 73.2}
        D_solution = growth_factor_D(z, pars)
        plt.plot(
            z,
            D_solution,
            linestyle="-",
            color=f"C{i}",
            linewidth=2,
            label=f"$w_0$ = {W_0_list[i]}; $w_a$ = {W_a_list[i]}",
        )
    plt.xlabel(f"$z$")
    plt.ylabel(f"$D_+(z)$")
    plt.xscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "/home/etudiant15/Documents/STAGE CPPM/Figures/growth_factor_D.pdf",
        bbox_inches="tight",
    )
    plt.show()


def plot_f():  # f pour différentes valeurs de w_0 et w_a, en fonction de z, pour Omega_Lambda fixé
    a = 10.0 ** np.linspace(-2, 0, 100)  # de 10**-2 à 10**0
    z = 1 / a - 1
    plt.figure()
    W_0_list = [-1, -0.8, -0.6, -0.4, -0.2]
    W_a_list = [0, -0.6, -1.2, -1.8, -2.4]
    # Omega_m = 0.3
    Omega_m = 0.3
    for i in range(len(W_a_list)):
        pars = {"Omega_m": Omega_m, "W_0": W_0_list[i], "W_a": W_a_list[i], "H_0": 73.2}
        f_solution = growth_rate_f(z, pars)
        # f_values = f_solution[:,0]
        plt.plot(
            z,
            f_solution,
            #'o',
            linestyle="-",
            color=f"C{i}",
            # linewidth=2,
            label=f"$w_0$ = {W_0_list[i]}; $w_a$ = {W_a_list[i]}",
        )
    plt.xlim(10**-2, 10**1)
    plt.ylim(0.4, 1.1)
    plt.xlabel(f"$z$")
    plt.ylabel(f"$f(z)$")
    plt.xscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "/home/etudiant15/Documents/STAGE CPPM/Figures/growth_rate_f.pdf",
        bbox_inches="tight",
    )
    plt.show()


def plot_H_z_times_1plusz():  # derivée de a pour différentes valeurs de w_0 et w_a, Omega_Lambda fixé, en fonction de z
    a = 10.0 ** np.linspace(-2, 0, 100)  # de 10**-2 à 10**0
    z = 1 / a - 1
    H_vec = np.vectorize(H)
    plt.figure()
    W_0_list = [-1, -0.8, -0.6, -0.4, -0.2]
    W_a_list = [0, -0.6, -1.2, -1.8, -2.4]
    # Omega_m = 0.3
    Omega_m = 0.3
    for i in range(len(W_a_list)):
        pars = {"Omega_m": Omega_m, "W_0": W_0_list[i], "W_a": W_a_list[i], "H_0": 73.2}
        plt.plot(
            z,
            H_vec(a, pars) * a,
            linestyle="-",
            color=f"C{i}",
            linewidth=2,
            label=f"$w_0$ = {W_0_list[i]}; $w_a$ = {W_a_list[i]}",
        )
    plt.xlabel(f"$z$")
    plt.ylabel("$H(z)(1+z)[km/s/Mpc]$")
    plt.xlim(10**-2, 10**1)
    plt.ylim(50, 130)
    plt.xscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "/home/etudiant15/Documents/STAGE CPPM/Figures/H_z_times_1plusz.pdf",
        bbox_inches="tight",
    )
    plt.show()
 

def plot_D_over_a():  # D/a pour différentes valeurs de W et Omega_Lambda, en fonction de a
    a = 10.0 ** np.linspace(-2, 0, 1000)  # de 10**-2 à 10**0
    z = 1 / a - 1
    plt.figure()
    W_0_list = [-1, -1, -0.5, 0]
    W_a_list = [0, 0, 0, 0]
    Omega_Lambda_list = [0.69, 0.72, 0.69, 0]
    for i in range(len(Omega_Lambda_list)):
        pars = {
            "Omega_m": 1 - Omega_Lambda_list[i],
            "Omega_Lambda": Omega_Lambda_list[i],
            "W_0": W_0_list[i],
            "W_a": W_a_list[i],
            "H_0": 73.2,
        }
        plt.plot(
            a,
            growth_factor_D(z, pars) / a,
            linestyle="-",
            color=f"C{i}",
            linewidth=2,
            label=f"$W$ = {W_0_list[i]}; $\Omega_\Lambda$ = {Omega_Lambda_list[i]}",
        )
    plt.xlabel("Scale factor a")
    plt.ylabel("Growth factor divided by a")
    plt.xscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "/home/etudiant15/Documents/STAGE CPPM/Figures/D_over_a.pdf",
        bbox_inches="tight",
    )
    plt.show()


def plot_f_times_Dplus():
    a = 10.0 ** np.linspace(-2, 0, 1000)  # de 10**-2 à 10**0
    z = 1 / a - 1
    plt.figure()
    W_0_list = [-1, -0.8, -0.6, -0.4, -0.2]
    W_a_list = [0, -0.6, -1.2, -1.8, -2.4]
    # Omega_m = 0.3
    Omega_m = 0.3
    for i in range(len(W_a_list)):
        pars = {"Omega_m": Omega_m, "W_0": W_0_list[i], "W_a": W_a_list[i], "H_0": 73.2}
        f_solution = growth_rate_f(z, pars)
        # f_values = f_solution[:,0]
        plt.plot(
            z,
            f_solution * growth_factor_D(z, pars),
            linestyle="-",
            color=f"C{i}",
            linewidth=2,
            label=f"$w_0$ = {W_0_list[i]}; $w_a$ = {W_a_list[i]}",
        )
    plt.xlabel("$z$")
    plt.ylabel(r"$f \times D_+(z)$")
    plt.xscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "/home/etudiant15/Documents/STAGE CPPM/Figures/f_times_D.pdf",
        bbox_inches="tight",
    )
    plt.show()


# DISTANCES

# ATTENTION ici on ne néglige pas omega_r
# omega _ m doit être dans le dictionnaire

Omega_r = 0.0001

"""
def H_sans_H0(z, pars): # on sort le H_0
    Omega_Lambda = pars['Omega_Lambda']
    Omega_m = pars['Omega_m']
    W_0 = pars['W_0']
    W_a = pars['W_a']
    a = 1/(1+z)
    hubble_rate = np.sqrt(Omega_m * a**-3 + Omega_r * a**-4 + Omega_Lambda*a**(-3*(1 + W_0 + W_a))*np.exp(-3*W_a*(1-a)))
    return hubble_rate
"""


def khi(z, pars):
    def invH(z_prime, pars):
        a = 1 / (1 + z_prime)
        return 1 / H(a, pars)

    res, err = quad(invH, 0, z, pars)
    coeff = 3 * 10**5
    return res * coeff


def d_A(z, pars):
    a = 1 / (1 + z)
    return a * khi(z, pars)


def d_L(z, pars):
    a = 1 / (1 + z)
    return khi(z, pars) / a


def plot_alldistances():  # toutes les distances sur le même graphique
    plt.figure()
    a = 10.0 ** np.linspace(-2, 0, 1000)  # de 10**-2 à 10**0
    z = 1 / a - 1
    W_0_list = [-1, -0.8, -0.6, -0.4, -0.2]
    W_a_list = [0, -0.6, -1.2, -1.8, -2.4]
    Omega_m_list = [0.1, 0.3, 0.9]
    for i in range(len(Omega_m_list)):
        pars = {
            "Omega_m": Omega_m_list[i],
            "Omega_Lambda": 1 - Omega_m_list[i] - Omega_r,
            "W_0": W_0_list[i],
            "W_a": W_a_list[i],
            "H_0": 73.2,
        }
        khi_values = [khi(z_i, pars) for z_i in z]
        d_A_values = [d_A(z_i, pars) for z_i in z]
        d_L_values = [d_L(z_i, pars) for z_i in z]
        plt.plot(
            z,
            khi_values,
            linestyle="-",
            color=f"C{i}",
            linewidth=2,
            label=f'$\chi$; $\Omega_m$ = {pars["Omega_m"]}; $\Omega_\lambda$ = {pars["Omega_Lambda"]:.2f}; $w_0$ = {W_0_list[i]}; $w_a$ = {W_a_list[i]}',
        )
        plt.plot(
            z,
            d_A_values,
            linestyle="--",
            color=f"C{i}",
            linewidth=2,
            label=f'$d_A$; $\Omega_m$ = {pars["Omega_m"]}; $\Omega_\lambda$ = {pars["Omega_Lambda"]:.2f}; $w_0$ = {W_0_list[i]}; $w_a$ = {W_a_list[i]}',
        )
        plt.plot(
            z,
            d_L_values,
            linestyle="-.",
            color=f"C{i}",
            linewidth=2,
            label=f'$d_L$; $\Omega_m$ = {pars["Omega_m"]}; $\Omega_\lambda$ = {pars["Omega_Lambda"]:.2f}; $w_0$ = {W_0_list[i]}; $w_a$ = {W_a_list[i]}',
        )
    plt.xlabel("$z$")
    plt.ylabel("Distance [$h^{-1}$ Mpc]")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "/home/etudiant15/Documents/STAGE CPPM/Figures/distances.pdf",
        bbox_inches="tight",
    )
    plt.show()


def plot_comoving_distance():  # khi
    plt.figure()
    a = 10.0 ** np.linspace(-2, 0, 1000)  # de 10**-2 à 10**0
    z = 1 / a - 1
    W_0_list = [-1, -0.8, -0.6, -0.4, -0.2]
    W_a_list = [0, -0.6, -1.2, -1.8, -2.4]
    Omega_m_list = [0.1, 0.3, 0.9]
    for i in range(len(Omega_m_list)):
        pars = {
            "Omega_m": Omega_m_list[i],
            "Omega_Lambda": 1 - Omega_m_list[i] - Omega_r,
            "W_0": W_0_list[i],
            "W_a": W_a_list[i],
            "H_0": 73.2,
        }
        khi_values = [khi(z_i, pars) for z_i in z]
        plt.plot(
            z,
            khi_values,
            linestyle="-",
            color=f"C{i}",
            linewidth=2,
            label=f'$\chi$; $\Omega_m$ = {pars["Omega_m"]}; $\Omega_\lambda$ = {pars["Omega_Lambda"]}; $w_0$ = {W_0_list[i]}; $w_a$ = {W_a_list[i]}',
        )
    plt.xlabel("$z$")
    plt.ylabel("Distance $\chi$ [$h^{-1}$ Mpc]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_angular_diameter_distance():  # d_A
    plt.figure()
    a = 10.0 ** np.linspace(-2, 0, 1000)  # de 10**-2 à 10**0
    z = 1 / a - 1
    W_0_list = [-1, -0.8, -0.6, -0.4, -0.2]
    W_a_list = [0, -0.6, -1.2, -1.8, -2.4]
    Omega_m_list = [0.1, 0.3, 0.9]
    for i in range(len(Omega_m_list)):
        pars = {
            "Omega_m": Omega_m_list[i],
            "Omega_Lambda": 1 - Omega_m_list[i] - Omega_r,
            "W_0": W_0_list[i],
            "W_a": W_a_list[i],
            "H_0": 73.2,
        }
        d_A_values = [d_A(z_i, pars) for z_i in z]
        plt.plot(
            z,
            d_A_values,
            linestyle="--",
            color=f"C{i}",
            linewidth=2,
            label=f'$d_A$; $\Omega_m$ = {pars["Omega_m"]}; $\Omega_\lambda$ = {pars["Omega_Lambda"]}; $w_0$ = {W_0_list[i]}; $w_a$ = {W_a_list[i]}',
        )
    plt.xlabel("$z$")
    plt.ylabel("Distance $d_A$ [$h^{-1}$ Mpc]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_luminosity_distance():  # d_L
    plt.figure()
    a = 10.0 ** np.linspace(-2, 0, 1000)  # de 10**-2 à 10**0
    z = 1 / a - 1
    W_0_list = [-1, -0.8, -0.6, -0.4, -0.2]
    W_a_list = [0, -0.6, -1.2, -1.8, -2.4]
    Omega_m_list = [0.1, 0.3, 0.9]
    for i in range(len(Omega_m_list)):
        pars = {
            "Omega_m": Omega_m_list[i],
            "Omega_Lambda": 1 - Omega_m_list[i] - Omega_r,
            "W_0": W_0_list[i],
            "W_a": W_a_list[i],
            "H_0": 73.2,
        }
        d_L_values = [d_L(z_i, pars) for z_i in z]
        plt.plot(
            z,
            d_L_values,
            linestyle="-.",
            color=f"C{i}",
            linewidth=2,
            label=f'$d_L$; $\Omega_m$ = {pars["Omega_m"]}; $\Omega_\lambda$ = {pars["Omega_Lambda"]}; ; $w_0$ = {W_0_list[i]}; $w_a$ = {W_a_list[i]}',
        )
    plt.xlabel("$z$")
    plt.ylabel("Distance $d_L$ [$h^{-1}$ Mpc]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def Dv_over_rd(z_val, pars):
    a = 1 / (1 + z_val)
    D_A = d_A(z_val, pars) * pars["H_0"]
    D_M = D_A * (1 + z_val)
    H_val = H(a, pars) / pars["H_0"]
    D_H = c / H_val
    Dv = (z_val * D_M**2 * D_H) ** (1 / 3)
    return Dv / pars["H_0xr_d"]

"""def Dv_over_rd(z_val, pars):
    try:
        a = 1 / (1 + z_val)
        D_A = d_A(z_val, pars) * pars["H_0"]
        D_M = D_A * (1 + z_val)
        H_val = H(a, pars) / pars["H_0"]
        
        # Protection
        if H_val <= 0:
            return np.nan
            
        D_H = c / H_val
        Dv = (z_val * D_M**2 * D_H) ** (1 / 3)
        
        if pars["H_0xr_d"] <= 0:
            return np.nan
            
        return Dv / pars["H_0xr_d"]
        
    except:
        return np.nan"""


def DM_over_DH(z_val, pars):
    a = 1 / (1 + z_val)
    D_M = d_A(z_val, pars) * (1 + z_val)
    D_H = c / H(a, pars)
    return D_M / D_H


"""def DM_over_DH(z_val, pars):
    try:
        a = 1 / (1 + z_val)
        D_M = d_A(z_val, pars) * (1 + z_val)
        H_val = H(a, pars)
        
        # Protection
        if H_val <= 0:
            return np.nan
            
        D_H = c / H_val
        return D_M / D_H
        
    except:
        return np.nan"""