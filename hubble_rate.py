import matplotlib.pyplot as plt
import numpy as np

H_0 = 73.2 #km s mpc


def hubble_rate(a, omega_m, omega_r, omega_lambda):
    hubble_rate= H_0*np.sqrt(omega_m * a**-3 + omega_r * a**-4 + omega_lambda)
    return hubble_rate


omega_r = 0.0001

a_min = 10**-5
a_max = 1
a_values = np.linspace(a_min, a_max, 1000)


omega_m = [0.1, 0.3, 0.9]


for elt in omega_m:
    omega_lambda = 1 - elt #univers plat
    H = hubble_rate(a_values, elt, omega_r, omega_lambda)
    plt.plot(a_values, H, 
        label = f'$\Omega_m$ = {elt}; $\Omega_r$ = {omega_r}; $\Omega_\lambda$ = {omega_lambda}')

plt.title("Hubble rate as a function of a", fontsize=14)
plt.xlabel("a")
plt.ylabel("H(t)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


#fontsize=12 pour la taille



