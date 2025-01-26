#%%
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from HSGP import se_spectral_density_1D
import matplotlib.pyplot as plt
import scipy.stats as sp

path = '5_Shape_constrained_modelling'

plt.rcParams.update({
    'font.size': 15.0,
    'axes.titlesize': 'medium'
})
textwidth = 5.9
plotwidth = 1.5*0.95*textwidth
plotheight = plotwidth*0.8


def eigenvalues(L,i):
    return i*np.pi/(2*L)


def monotonic_eigenfuctions(x,L,i,j):
    gamma_plus = eigenvalues(L,i) + eigenvalues(L,j) 
    gamma_minus = eigenvalues(L,i) - eigenvalues(L,j) 
    if x > L:
        x = L
    elif x < -L:
        x = -L

    if i == j:
        return (x+L)/(2*L) - np.sin(gamma_plus*(x+L))/(2*gamma_plus*L)
    else:
        return np.sin(gamma_minus*(x+L))/(2*gamma_minus*L) - np.sin(gamma_plus*(x+L))/(2*gamma_plus*L)
    

def ushaped_eigenfuctions(x,L,i,j):
    gamma_plus = eigenvalues(L,i) + eigenvalues(L,j) 
    gamma_minus = eigenvalues(L,i) - eigenvalues(L,j)
    Fprime = 0

    if x > L:
        if i == j:
            Fprime = 1
    
    if x < -L:
        x = -L
    
    if i == j:
        return (x+L)*(x+L)/(4*L) + (np.cos(gamma_plus*(x+L)) - 1)/(2*L*gamma_plus**2) + Fprime*(x-L)
    else:
        return (1-np.cos(gamma_minus*(x+L)))/(2*L*gamma_minus**2) - (np.cos(gamma_plus*(x+L))-1)/(2*L*gamma_plus**2)  + Fprime*(x-L)

#%%
num_samples = 20
ms = [2,4,16]
Ls = [5,10,20]

kappa = 1
scale = 1
num_x = 20
x_grid = np.linspace(-10, 10, num_x)

np.random.seed(1)

fig, ax = plt.subplots(3,3, figsize=(plotwidth, plotheight))
colors_m = ['deeppink', 'orange', 'red']
colors_L = ['cyan', 'blue', 'indigo']

f_0_monotonic = sp.t.rvs(4, loc=0, scale=1, size=(num_samples))
z = np.random.normal(0, 1, (ms[-1], num_samples))

for k, m in enumerate(ms):
    for l,L in enumerate(Ls):
        lambdas = np.zeros(m)
        for i in range(m):
            lambdas[i] = eigenvalues(L,i)
        alpha_scale = se_spectral_density_1D(lambdas,[kappa,scale])[:,None]
        alpha = z[:m,:]*alpha_scale
        monotonic_eig = np.zeros((m, m))
        monotonic_samples = np.zeros((num_samples, num_x))

        for s in range(num_samples):
            for t,x in enumerate(x_grid):
                for i in range(1,m+1):
                    for j in range(1,m+1):
                        monotonic_eig[i-1,j-1] = monotonic_eigenfuctions(x, L, i, j)
                monotonic_samples[s, t] = alpha[:,s].T@monotonic_eig@alpha[:,s]

        monotonic_samples += f_0_monotonic[:, None]


        ax[k,l].plot(x_grid, monotonic_samples.T, alpha = 0.2, color=colors_m[k])
        ax[k,l].plot(x_grid, monotonic_samples.T, alpha = 0.15, color=colors_L[l])
        # ax[k,l].set_title(f'm = {m}, L = {L}')

        if k == 0:
            ax[k,l].text(0.5, 1.1, f'L = {L}', transform=ax[k,l].transAxes, ha='center')
        if l == 0:
            ax[k,l].text(-0.25, 0.5, f'm = {m}', transform=ax[k,l].transAxes, va='center', rotation=90)

        ax[k,l].tick_params(axis='y', labelrotation=90)  # Rotate y-tick labels

fig.tight_layout()
plt.savefig('5_Shape_constrained_modelling/monotonic_prior_m_L.png', bbox_inches='tight', dpi=200)
plt.show()


#%%

num_x = 20
x_grid = np.linspace(-10, 10, num_x)

ms = [2,4,16]
Ls = [5,10,20]

kappa = 1
scale = 1

F_0_ushaped = sp.t.rvs(4, loc=10, scale=1, size=(num_samples))
f_0_ushaped = -sp.gamma.rvs(2, loc=0, scale=5, size=(num_samples))
z = np.random.normal(1, 1, (ms[-1], num_samples))

fig, ax = plt.subplots(3,3, figsize=(plotwidth, plotheight))
colors_m = ['deeppink', 'orange', 'red']
colors_L = ['cyan', 'blue', 'indigo']

for k,m in enumerate(ms):
    for l,L in enumerate(Ls):
        lambdas = np.zeros(m)
        for i in range(m):
            lambdas[i] = eigenvalues(L,i)
        alpha_scale = se_spectral_density_1D(lambdas,[kappa,scale])[:,None]
        alpha = z[:m,:]*alpha_scale
        ushaped_eig = np.zeros((m, m))

        ushaped_samples = np.zeros((num_samples, num_x))

        for s in range(num_samples):
            for t,x in enumerate(x_grid):
                for i in range(1,m+1):
                    for j in range(1,m+1):
                        ushaped_eig[i-1,j-1] = ushaped_eigenfuctions(x, L, i, j)
                ushaped_samples[s, t] = alpha[:,s].T@ushaped_eig@alpha[:,s]
            
            ushaped_samples[s, :] += f_0_ushaped[s]*(x_grid + L)
        ushaped_samples += F_0_ushaped[:, None]
    
        ax[k,l].plot(x_grid, ushaped_samples.T, alpha = 0.2, color=colors_m[k])
        ax[k,l].plot(x_grid, ushaped_samples.T, alpha = 0.15, color=colors_L[l])
        if k == 0:
            ax[k,l].text(0.5, 1.1, f'L = {L}', transform=ax[k,l].transAxes, ha='center')
        if l == 0:
            ax[k,l].text(-0.25, 0.5, f'm = {m}', transform=ax[k,l].transAxes, va='center', rotation=90)

        ax[k,l].set_yticks([0.0, ax[k,l].get_yticks()[-2]])  # Display only first and last yticks
        ax[k,l].tick_params(axis='y', labelrotation=90)  # Rotate y-tick labels

                
plt.savefig('5_Shape_constrained_modelling/ushaped_prior_m_L.png', bbox_inches='tight', dpi=200)
fig.tight_layout()
plt.show()
  
#%% ### KAPPA AND f0 PLOTS
# %% plotting prior ushaped samples with different f_0 and kappa values
num_samples = 100
m = 16
L = 10

num_samples = 50
num_x = 100
x_grid = np.linspace(-L, L, num_x)
f_0_value_and_kappa = [[-10,0.5], [-0.01,0.01], [-0.01,2], [-10,1]]
scale = 1

fig, axs = plt.subplots(2, 2, figsize=(8, 6))
colors = ['cyan', 'red', 'orange', 'indigo']
subtitle = ['(a)', '(b)', '(c)', '(d)']

for idx, (f_0_value, kappa) in enumerate(f_0_value_and_kappa):
    lambdas = np.zeros(m)
    for i in range(m):
        lambdas[i] = eigenvalues(L, i)
    
    alpha_scale = se_spectral_density_1D(lambdas, [kappa, scale])
    alpha = np.random.normal(0, alpha_scale[:, None], (m, num_samples))
    
    F_0_ushaped = sp.t.rvs(4, loc=0, scale=1, size=(num_samples))
    f_0_ushaped = np.ones(num_samples) * f_0_value
    
    ushaped_eig = np.zeros((m, m))
    ushaped_samples = np.zeros((num_samples, num_x))

    
    for s in range(num_samples):
        for t, x in enumerate(x_grid):
            for i in range(1, m + 1):
                for j in range(1, m + 1):
                    ushaped_eig[i - 1, j - 1] = ushaped_eigenfuctions(x, L, i, j)
            ushaped_samples[s, t] = alpha[:, s].T @ ushaped_eig @ alpha[:, s]
        
        ushaped_samples[s, :] += f_0_ushaped[s] * (x_grid + L)
    ushaped_samples += F_0_ushaped[:, None]
    
    ax = axs[idx // 2, idx % 2]
    for i in range(50):
        ax.plot(x_grid, ushaped_samples[i, :], color=colors[idx], alpha=0.1)
    ax.set_xlabel('x')
    ax.set_ylabel('F(x)')
    ax.set_title(f'{subtitle[idx]}   $f_0$={f_0_value}, $\kappa$={kappa}')

fig.suptitle('Prior samples from u-shaped HS model with different $f_0$ and $\kappa$ values')
plt.tight_layout()
plt.show()
