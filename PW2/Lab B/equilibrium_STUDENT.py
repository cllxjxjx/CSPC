"""
PW2 Lab B Part 4 -- chemical equilibrium via the equilibrium constant K.

Reaction  H2 + I2 <=> 2 HI, starting from 1 mol H2 and 1 mol I2.
As the reaction proceeds by an extent x:  H2 = 1-x,  I2 = 1-x,  HI = 2x.
At equilibrium the composition satisfies the equilibrium constant
        K = [HI]^2 / ([H2][I2]) = (2x)^2 / ((1-x)(1-x)).
Given K, find the extent x. Solve it TWO ways and compare.
Run:  python equilibrium.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, minimize

K = 15.6
a = b = 1.0

def k_imbalance(x):
    return ((2 * x)**2) / ((1 - x) * (1 - x)) - K

x_newton = newton(k_imbalance, x0=0.5)

def obj_slsqp(x):
    return k_imbalance(x[0])**2

res_slsqp = minimize(obj_slsqp, x0=[0.5], method="SLSQP", bounds=[(0.001, 0.999)])
x_slsqp = res_slsqp.x[0]

print(f"Newton method x = {x_newton:.4f}")
print(f"SLSQP method x  = {x_slsqp:.4f}")


n_H2 = 1 - x_newton
n_I2 = 1 - x_newton
n_HI = 2 * x_newton

print(f"\nBalance:")
print(f"n(H2) = {n_H2:.2f} mol")
print(f"n(I2) = {n_I2:.2f} mol")
print(f"n(HI) = {n_HI:.2f} mol")

x_vals = np.linspace(0, 0.95, 200)
H2_vals = 1 - x_vals
I2_vals = 1 - x_vals
HI_vals = 2 * x_vals

plt.figure(figsize=(7, 5))
plt.plot(x_vals, H2_vals, 'r-', label='$H_2$ (reagent)')
plt.plot(x_vals, I2_vals, 'g--', label='$I_2$ (reagent)')
plt.plot(x_vals, HI_vals, 'b-', label='$HI$ (product)')


plt.axvline(x=x_newton, color='gray', linestyle=':', label=f'Balance (x ≈ {x_newton:.2f})')
plt.scatter([x_newton], [n_HI], color='blue', zorder=5)
plt.scatter([x_newton], [n_H2], color='red', zorder=5)

plt.xlabel('reaction progress')
plt.ylabel('amount (mol)')
plt.title('Chemical Equilibrium: $H_2 + I_2 \\rightleftharpoons 2HI$')
plt.legend()
plt.grid(True)

plt.savefig('equilibrium.png')
print("Graph saved as equilibrium.png.")
# TODO 1: write k_imbalance(x) = (2x)^2/((a-x)(b-x)) - K.
#         It equals zero exactly at equilibrium.

# TODO 2 (method 1): use scipy.optimize.newton to find the root of k_imbalance
#         (start x0=0.5). This is root-finding.

# TODO 3 (method 2): use scipy.optimize.minimize to minimise k_imbalance(x)**2
#         (method "SLSQP", bounds [(0, 0.999)], x0=[0.5]). Print both answers
#         and confirm they agree.

# TODO 4: report the equilibrium amounts (H2, I2, HI), and plot how the three
#         amounts change with the extent x, marking the equilibrium. Save
#         equilibrium.png.
