#%%
import os, sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from HSGP import GP_true, squared_exponential, se_spectral_density_1D
import numpy as np
import matplotlib.pyplot as plt



#%% PRIOR SAMPELS FROM PDF AND CDF

def eigenvalues(L,i):
    return i*np.pi/(2*L)

def positive_eigenfunction(j, x, L):
    return 1/np.sqrt(L)*np.sin(np.pi*j/(2*L)*(x+L))*(np.abs(x) <= L)

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

L = 5
m = 16
num_x = 100
x_grid = np.linspace(-L-1, L+1, num_x)
num_samples = 50

kappa = 1
scale = 1

lambdas = np.zeros(m)

for i in range(m):
    lambdas[i] = eigenvalues(L,i)

alpha_scale = se_spectral_density_1D(lambdas,[kappa,scale])

alpha = np.random.normal(0, alpha_scale[:, None], (m, num_samples))


Z = np.sum(alpha**2,axis=0)
pdf_eig = np.zeros((m, m))
pdf_samples = np.zeros((num_samples, num_x))
CDF_eig = np.zeros((m, m))
CDF_samples = np.zeros((num_samples, num_x))

for s in range(num_samples):
    for t,x in enumerate(x_grid):
        for i in range(1, m + 1):
            for j in range(1, m + 1):
                pdf_eig[i-1, j-1] = positive_eigenfunction(i,x,L)*positive_eigenfunction(j,x,L)
                CDF_eig[i-1, j-1] = monotonic_eigenfuctions(x, L, i, j)
        pdf_samples[s, t] = alpha[:, s].T @ pdf_eig @ alpha[:, s]
        CDF_samples[s, t] = alpha[:, s].T @ CDF_eig @ alpha[:, s]

pdf_samples /= Z[:,None]
CDF_samples /= Z[:,None]
#%% INTEGRABLE, POSITIVE PRIOR SAMPLES
fig, axs = plt.subplots(1, 2, figsize=(8, 4))

# Plot PDF samples
axs[0].set_title('Prior samples from positive model\n p(x) = 1/Z h(x)')
for i in range(num_samples):
    axs[0].plot(x_grid, pdf_samples[i, :], color='orange', alpha=0.1)
axs[0].set_xlabel('x')
axs[0].set_ylabel('p(x)')

# Plot CDF samples
axs[1].set_title('Prior samples from corresponding CDF \n CDF(x)$= \int_{-L}^{x}$ p(x)dx')
for i in range(num_samples):
    axs[1].plot(x_grid, CDF_samples[i, :], color='red', alpha=0.1)
axs[1].set_xlabel('x')
axs[1].set_ylabel('CDF(x)')

plt.tight_layout()
plt.show()



#%%
plt.rcParams.update({
    'font.size': 16.0,
    'axes.titlesize': 'medium'
})

textwidth = 5.9
plotscale = 0.95
plotwidth = textwidth*plotscale

theta = [0.1, 1.5, 1.1]


GP = GP_true(squared_exponential,theta)
GP.fit(np.zeros((0, 1)), np.zeros((0, 1)))

N_samples = 10000
N_x = 1000

X = np.linspace(-5,5,N_x)

samples = GP.generate_samples(X[:, None], N_samples).reshape(N_samples,N_x)


positive_model = samples**2
#%%
GPmean = np.mean(samples, axis=0)
GPvar = np.var(samples, axis=0)

positive_mean = np.mean(positive_model, axis=0)
positive_var = np.var(positive_model, axis=0)

theoretic_GP_mean = np.zeros(N_x)
theoretic_GP_var = theta[1]**2*np.ones(N_x)
theoretic_positive_mean = theta[1]**2*np.ones(N_x)
theoretic_positive_var = 2*theta[1]**4*np.ones(N_x)

print('mean squared error between theoretic and emperical values. mean:', np.mean((GPmean-theoretic_GP_mean)**2), 'variance:', np.mean((GPvar-theoretic_GP_var)**2))
print('True GP mean:', 0, 'True GP variance:', theta[1]**2)
print('mean squared error between theoretic and emperical values. mean:', np.mean((positive_mean-theoretic_positive_mean)**2), 'variance:', np.mean((positive_var-theoretic_positive_var)**2))
# print(f'Positive model mean: {positive_mean}, Positive model variance: {positive_var}')
print('True positive mean:', theta[1]**2, 'True positive variance:', 2*theta[1]**4)
#%%
plt.figure(figsize=(plotwidth, plotwidth*1/2))
plt.title('Positive model mean')
plt.plot(X, positive_mean,color='r', label='Emperical Mean')
plt.plot(X, theoretic_positive_mean, color='b',label='Theoretic Mean')
plt.plot(X, positive_model[::50].T, color='g', alpha=0.05)
plt.plot(X, positive_model[0], color='g', alpha=0.05, label='Samples')
plt.legend()
plt.xlabel('x')
plt.ylabel('h(x)')
plt.savefig('figures/positive_mean.png', bbox_inches='tight', dpi=200)
plt.show()

plt.figure(figsize=(plotwidth, plotwidth*1/2))
plt.title('Positive model variance')
plt.plot(X, positive_var,color='r', label='Emperical Variance')
plt.plot(X, theoretic_positive_var, color='b',label='Theoretic Variance')
plt.ylim(0, 40)
plt.xlabel('x')
plt.ylabel('Variance of h(x)')
plt.legend()
plt.savefig('figures/positive_variance.png', bbox_inches='tight', dpi=200)
plt.show()
#%%

m_mse = np.zeros(50)
v_mse= np.zeros(50)
m_rel = np.zeros(50)
v_rel= np.zeros(50)

n_grid = np.logspace(1, 4, num=50, base=10, dtype=int)

for i,n  in enumerate(n_grid):
    m_mse[i] = np.mean((np.mean(positive_model[:n], axis=0)-theoretic_positive_mean)**2)
    v_mse[i] = np.mean((np.var(positive_model[:n], axis=0)-theoretic_positive_var)**2)

    m_rel[i] = np.linalg.norm(np.mean(positive_model[:n], axis=0)-theoretic_positive_mean)/np.linalg.norm(theoretic_positive_mean)
    v_rel[i] = np.linalg.norm(np.var(positive_model[:n], axis=0)-theoretic_positive_var)/np.linalg.norm(theoretic_positive_var)



#%%
plt.figure(figsize=(plotwidth, 1/2*plotwidth))
plt.plot(n_grid, m_mse)
plt.xlabel('Number of samples')
plt.ylabel('MSE of empirical and theoretical Mean')
plt.title('Convergence of Mean')
plt.show()

plt.figure(figsize=(plotwidth, 1/2*plotwidth))
plt.plot(n_grid, v_mse)
plt.xlabel('Number of samples')
plt.ylabel('MSE of empirical and theoretical Variance')
plt.title('Convergence of Variance')
plt.tight_layout()
plt.show()
# %%
plt.figure(figsize=(plotwidth, 1/2*plotwidth))
plt.plot(n_grid, m_rel)
plt.xlabel('Number of samples')
plt.ylabel('relative error of mean')
plt.title('Convergence of Mean')
plt.savefig('figures/relative_convergence_of_positive_function_mean', dpi=200, bbox_inches='tight')
plt.show()

plt.figure(figsize=(plotwidth, 1/2*plotwidth))
plt.plot(n_grid, v_rel)
plt.xlabel('Number of samples')
plt.ylabel('relative error of variance')
plt.title('Convergence of Variance')
plt.savefig('figures/relative_convergence_of_positive_function_var', dpi=200, bbox_inches='tight')
plt.show()

# %%
