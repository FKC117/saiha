import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.power import TTestIndPower

power_analysis = TTestIndPower()
n_range = np.linspace(5, 100, 50).astype(int)
power = 0.8
alpha = 0.05

effects = []
for n_val in n_range:
    es = power_analysis.solve_power(nobs1=n_val, power=power, alpha=alpha)
    # Simulate the rare case where statsmodels returns an array
    if n_val == 50:
        es = np.array([es])
    effects.append(es)

try:
    plt.plot(n_range, effects)
    print("Plot successful")
except Exception as e:
    print(f"Plot failed: {e}")

# Fix
effects_fixed = [float(e) for e in effects]
plt.plot(n_range, effects_fixed)
print("Fixed plot successful")
