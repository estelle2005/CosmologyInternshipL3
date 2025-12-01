import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad


H_0 = 73.2
Omega_m = 0.3
Omega_Lambda = 1 - Omega_m

### $w(a) = w_0 + (1-a)w_a $
### Compare $f(a)$ for :\n",

    "- $w_0 = -1.0$ and $w_a = 0$ (cosmological constant)\n",
    "- $w_0 = -1.2$ and $w_a = 0$\n",
    "- $w_0 = -0.8$ and $w_a = 0$\n",
    "- $w_0 = -1.0$ and $w_a = 0.1$\n",
    "- $w_0 = -1.0$ and $w_a = -0.1$\n",

"### Hint: first find the equation for $H(a)$, then the equation for $H^\\prime$, then write a new function to solve the differential equation"


def H(a, W_0, W_a):
    return H_0 *np.sqrt(Omega_m * a**-3 + Omega_Lambda*a**(-3*(1 + W_0 + W_a))*np.exp(3*W_a*(1-a)))