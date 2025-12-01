import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constantes cosmologiques
H0 = 70  # km/s/Mpc
Omega = 0.3

# Fonction H(a) pour univers ΛCDM
def H(a):
    Omega_Lambda = 1 - Omega
    return H0 * np.sqrt(Omega / a**3 + Omega_Lambda)

# Coefficient a(a) = d(ln(a^3 H))/da
def a_coeff(a):
    dH_da = (H0**2 * -3*Omega/(2*a**4)) / H(a)  # dérivée de H(a)
    return 3/a + dH_da / H(a)

# Coefficient b(a) = 3 Omega H0^2 / (2 a^5 H^2)
def b_coeff(a):
    return 3 * Omega * H0**2 / (2 * a**5 * H(a)**2)

# Système du premier ordre
def dydA(a, yvec):
    y1, y2 = yvec
    dy1_da = y2
    dy2_da = - a_coeff(a) * y2 + b_coeff(a) * y1
    return [dy1_da, dy2_da]

# Conditions initiales
a0 = 0.01  # a au début
y0 = [a0, 0]  # y(a0) = a0, y'(a0)=0

# Intervalle d'intégration
a_span = (a0, 1.0)
a_eval = np.linspace(a0, 1.0, 500)

# Résolution numérique
sol = solve_ivp(dydA, a_span, y0, t_eval=a_eval, method='RK45')

# Tracer la solution
plt.plot(sol.t, sol.y[0])
plt.xlabel("a")
plt.ylabel("y(a)")
plt.title("Solution numérique de l'équation différentielle")
plt.grid(True)
plt.show()
