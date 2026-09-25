"""
PW2 Lab A -- Motion from tracking data.

Read noisy free-fall position measurements, then:
  - differentiate once  -> velocity
  - differentiate twice -> acceleration (should be ~ constant -g, but noisy!)
  - integrate the acceleration back up -> recover velocity and position

Complete the TODOs. Run:  python analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

data = np.loadtxt('freefall.csv',delimiter = ',',skiprows = 1)
t = data[:,0]
y = data[:,1]
v = np.gradient(y,t)
a = np.gradient(v,t)
mean_a = np.mean(a)
std_a = np.std(a)
print("The mean of acceleration is " + str(mean_a))
print("The mean of acceleration is " + str(std_a))
#it is noisy.

v_rec = cumulative_trapezoid(a, t, initial=0) + v[0]
y_rec = cumulative_trapezoid(v, t, initial=0) + y[0]
max_diff = np.max(np.abs(y-y_rec))
print("The maximum difference is " + str(max_diff))

fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize = (8,10),sharex = True)
ax1.plot(t,y,label = "position",color = 'blue')
ax1.set_ylabel("Position (m)")
ax1.set_title("Motion from data")
ax1.legend()
ax1.grid(True)

ax2.plot(t,v,label = "velocity",color = 'red')
ax2.set_ylabel("Velocity (m/s)")
ax2.legend()
ax2.grid(True)

ax3.plot(t,a,label = "acceleration",color = 'green')
ax3.axhline(-9.81,color = 'red',linestyle = "--",label = 'True value -9.81')
ax3.set_ylabel("Acceleration (m/s*s)")
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.savefig("motion.png")



