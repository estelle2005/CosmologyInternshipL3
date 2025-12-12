import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

H_0 = 73.2
omega_r = 0.0001
coeff = 3 * 10**3


def H(z, omega_m, omega_r, omega_lambda):  # on sort le H_0
    a = 1 / (1 + z)
    hubble_rate = np.sqrt(omega_m * a**-3 + omega_r * a**-4 + omega_lambda)
    return hubble_rate


def invH(z_prime, omega_m, omega_r, omega_lambda):
    return 1 / H(z_prime, omega_m, omega_r, omega_lambda)


def khi(z, omega_m, omega_r, omega_lambda):
    res, err = quad(invH, 0, z, args=(omega_m, omega_r, omega_lambda))
    return res * coeff


def d_A(z, omega_m, omega_r, omega_lambda):
    a = 1 / (1 + z)
    return a * khi(z, omega_m, omega_r, omega_lambda)


def d_L(z, omega_m, omega_r, omega_lambda):
    a = 1 / (1 + z)
    return khi(z, omega_m, omega_r, omega_lambda) / a


z_values = np.linspace(0, 10, 1000)
omega_m_list = [0.1, 0.3, 0.9]


for i, om in enumerate(omega_m_list):
    omega_lambda = 1 - om  # univers plat
    parameters = {"omega_m": om, "omega_r": omega_r, "omega_lambda": omega_lambda}
    khi_values = [khi(z, **parameters) for z in z_values]
    d_A_values = [d_A(z, **parameters) for z in z_values]
    d_L_values = [d_L(z, **parameters) for z in z_values]
    # H = H(z_values, om, omega_r, omega_lambda)
    plt.plot(
        z_values,
        khi_values,
        linestyle="-",
        color=f"C{i}",
        linewidth=2,
        label=f"$\chi$; $\Omega_m$ = {om}; $\Omega_\lambda$ = {omega_lambda}",
    )
    plt.plot(
        z_values,
        d_A_values,
        linestyle="--",
        color=f"C{i}",
        linewidth=2,
        label=f"$d_A$; $\Omega_m$ = {om}; $\Omega_\lambda$ = {omega_lambda}",
    )
    plt.plot(
        z_values,
        d_L_values,
        linestyle="-.",
        color=f"C{i}",
        linewidth=2,
        label=f"$d_L$; $\Omega_m$ = {om}; $\Omega_\lambda$ = {omega_lambda}",
    )


plt.xlabel("$z$")
plt.ylabel("Distance [$h^{-1}$ Mpc]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
