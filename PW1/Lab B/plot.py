"""
PW1 Lab B -- read observed decay data and compare it to the analytical law.
Produce a 1x2 figure:  left = observed data,  right = analytical N0*exp(-lam*t),
with SHARED axes so the two shapes are directly comparable.

Complete the TODOs below. Run with:  python plot.py
"""

import numpy as np
import matplotlib.pyplot as plt

LAMBDA = 0.3     # decay constant, given

t,observed = np.loadtxt('decay_observed.csv',delimiter = ',',skiprows = 1,unpack = True)

N0 = observed[0]
analytical = N0*np.exp(-LAMBDA*t)

fig,(ax1,ax2) = plt.subplots(1,2,sharex = True,sharey = True,figsize = (10,4))
ax1.scatter(t,observed,color = 'blue',alpha = 0.7)
ax1.set_title("Observed Data")
ax1.set_xlabel("Time (t)")
ax1.set_ylabel("Counts (N)")

ax2.plot(t,analytical,color = 'red')
ax2.set_title("Analytical")
ax2.set_xlabel("Time (t)")

plt.tight_layout()
plt.savefig("figure.png")

