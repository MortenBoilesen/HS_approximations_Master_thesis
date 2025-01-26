#%%
import numpy as np
from matplotlib import pyplot as plt
np.random.seed(3)
x = np.linspace(-5,5, 15)

def benchmark_ushaped(idx, x):
    if idx == 0:
        return 2*np.ones(x.shape)
    elif idx == 1:
        return 0.1*(x-2)**2
    elif idx == 2:
        return 0.25*x**2
    elif idx == 3:
        return np.abs(x)
    elif idx == 4:
        return -2*np.sin( (x - x.min()) / (x.max() -  x.min())*2*np.pi - np.pi/2 ) +2
    elif idx == 5:
        y = 4*np.ones(np.shape(x))
        y[np.abs(x) <= 3] = 0
        return y

sigma = 1
fig, ax = plt.subplots(2, 3)

f = benchmark_ushaped(0, x)
y = f + np.random.normal(0, sigma, f.shape)

for i in range(6):
    row = i // 3
    col = i % 3
    f = benchmark_ushaped(i, x)
    y = f + np.random.normal(0, sigma, f.shape)

    ax[row, col].plot(x, f)
    ax[row, col].scatter(x, y, color='tab:orange')
    ax[row, col].set_title(f'Function {i}')

plt.tight_layout()
plt.show()
