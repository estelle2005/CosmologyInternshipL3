import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad  # Module d'intégration "quad"

H_0 = 73.2

omega_m = 0.3
omega_r = 0.0001
omega_lambda = 0.7

#h = H_0 / 100
coeff= 3*10**3

def H(a, omega_m, omega_r, omega_lambda): # on sort le H_0
    hubble_rate = np.sqrt(omega_m * a**-3 + omega_r * a**-4 + omega_lambda)
    return hubble_rate

def function(z_prime):
    a = 1/(1+z_prime)
    return 1/H(a, omega_m, omega_r, omega_lambda)

def khi(z): #normalement c'est fonction de t mais t n'est pas ds l'expression et on trace khi(z) ??
    res, err = quad(function, 0, z)
    return res * coeff

def d_A(z):
    a = 1 / (1+z)
    return a * khi(z)

def d_L(z) :
    a = 1 / (1+z)
    return khi(z)/a


z_min = 0
z_max = 10
z_values = np.linspace(z_min, z_max, 1000)

khi_values = [khi(z) for z in z_values]
d_A_values = [d_A(z) for z in z_values]
d_L_values = [d_L(z) for z in z_values]


plt.plot(z_values, khi_values, 
        linestyle='-', linewidth=2, label=f'$\chi$')
plt.plot(z_values, d_A_values, 
        linestyle='--', linewidth=2, label=f'$d_A$')
plt.plot(z_values, d_L_values, 
        linestyle='-.', linewidth=2, label=f'$d_L$')

plt.xlabel("$z$")
plt.ylabel("Distance [$h^{-1}$ Mpc]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
