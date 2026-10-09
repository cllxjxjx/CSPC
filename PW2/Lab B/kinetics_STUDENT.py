import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

data = np.genfromtxt('kinetics.csv', delimiter=',', skip_header=1)
t_data = data[:, 0]
C_measured = data[:, 1]


C0 = C_measured[0]


def total_error(k):
    k_val = k[0] if isinstance(k, (list, np.ndarray)) else k
    C_model = C0 * np.exp(-k_val * t_data)
    return np.sum((C_measured - C_model) ** 2)


res = minimize(total_error, x0=[0.5], method="SLSQP", bounds=[(0, 5)])
fitted_k = res.x[0]

print(f"Fitted rate constant k = {fitted_k:.4f}")


plt.figure(figsize=(7, 5))
plt.scatter(t_data, C_measured, color='red', label='Measured data')

t_smooth = np.linspace(t_data.min(), t_data.max(), 100)
C_smooth = C0 * np.exp(-fitted_k * t_smooth)

plt.plot(t_smooth, C_smooth, 'b-', label=f'Model curve (k = {fitted_k:.4f})')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Kinetics Fit')
plt.legend()
plt.grid(True)

plt.savefig('kinetics.png')
print("Graphic is saved as 'kinetics.png'.")