#%%
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({
    'font.size': 16.0,
    'axes.titlesize': 'medium'
})

textwidth = 5.9

def sqrt_eigenval(i, L):
    return np.pi * i / 2 / L

def eigenfun(x, i, L):
    return 1 / np.sqrt(L) * np.sin(sqrt_eigenval(i, L) * (x + L))

def eigenfun_2D(x, y, i, j, L):
    return eigenfun(x, i, L) * eigenfun(y, j, L)

L = 5

x = np.linspace(-L, L, 100)
y = np.linspace(-L, L, 100)
X, Y = np.meshgrid(x, y)

scale = 0.95*2.5
plotwidth= scale*textwidth
fig = plt.figure(figsize=(plotwidth, plotwidth ))

# 1D plot
ax1 = fig.add_subplot(4, 4, (2, 3))
for i in range(1, 5):
    ax1.plot(x, eigenfun(x, i, L), label=f'$j_k = {i}$')
ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
ax1.set_xlabel('x')
ax1.set_ylabel('$\phi(x)$')
ax1.set_title('Plot of $\phi_{j_k}$')

combs = [
        [1,1],
        [1,3],
        [4,2],
        [4,1]]

# Surface plots
for c, comb in enumerate(combs):
        i,j = comb
        n = c//2
        m = c % 2

        Z = eigenfun_2D(X, Y, i, j, L)
        ax = fig.add_subplot(4, 4, ((n) * 2 + m + 5), projection='3d')

        # ax = fig.add_subplot(4, 4, ((i - 1) * 2 + j + 4), projection='3d')
        ax.plot_surface(X, Y, Z, cmap='hot')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$\phi(x,y)$')
        ax.set_title(f'$\phi$ for $j_1 = {i}$, $j_2={j}$')

plt.suptitle('The first $m^d=2^2$ eigenfunctions of the negative Laplacian with Dirichlet boundary conditions.')
plt.tight_layout()
plt.savefig('eigenfun_togehter.png', bbox_inches='tight')
plt.show()
