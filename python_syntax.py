import numpy as np
import matplotlib.pylab as plt

x = np.arange(start=0.0, stop=10.0, step=0.1)
n = len(x)
K = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        K[i, j] = np.exp(-0.5 * (x[i] - x[j])**2)

y = np.random.multivariate_normal(np.zeros(n), K)

plt.plot(x, y)
plt.show()