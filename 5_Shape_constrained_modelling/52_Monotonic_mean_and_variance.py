#%%
import os, sys


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from HSGP import GP_true, squared_exponential
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.integrate import quad

plt.rcParams.update({
    'font.size': 16.0,
    'axes.titlesize': 'medium'
})
textwidth = 5.9
plotscale = 0.95
plotwidth = textwidth*plotscale

#%%

#sample from GP
theta = [0.1, 1.5, 1.1]

N_samples = 10000
N_x = 1000
L = 5
f0 = -2*L*theta[1]**2/2
F0 = 0

X = np.linspace(-L,L,N_x)
#%%
def positive_func(x):
    return x**2

try: 
    monotonic_model = np.load(f'monotonic_theta={theta}.npy'.replace(', ','-'))
    
except:
    print('Generating data')
    GP = GP_true(squared_exponential,theta)
    GP.fit(np.zeros((0, 1)), np.zeros((0, 1)))

    dx = 2*L/N_x
    samples = GP.generate_samples(X[:, None], N_samples).reshape(N_samples,N_x)


    # model the u-shaped function
    monotonic_model = np.zeros(samples.shape)

    positive_model = samples**2
    monotonic_model[:,0] = f0
    for i in range(1,N_x):
        monotonic_model[:,i] = f0 + np.trapz(positive_model[:,:i], dx=dx, axis=1)
        
    np.save(f'monotonic_theta={theta}'.replace(', ','-'),monotonic_model)

#%%
#calculate the mean and variance of the monotonic model and the theoretical values
monotonic_model_mean = np.mean(monotonic_model, axis=0)
monotonic_model_var = np.var(monotonic_model, axis=0)

z = (X+L)/theta[2]
theoretic_monotonic_model_mean = f0 + theta[1]**2*(X+L)
theoretic_monotonic_model_var = 2*theta[1]**4*theta[2]**2*(np.exp(-(z**2)) + np.sqrt(np.pi)*z*erf(z)-1)

print('MSE between theoretic and emperical mean:', np.mean((monotonic_model_mean-theoretic_monotonic_model_mean)**2))
print('MSE between theoretic and emperical variance:', np.mean((monotonic_model_var-theoretic_monotonic_model_var)**2))

print('Relative error between theoretic and emperical mean:', np.linalg.norm(monotonic_model_mean-theoretic_monotonic_model_mean)/np.linalg.norm(theoretic_monotonic_model_mean))
print('Relative error between theoretic and emperical variance:', np.linalg.norm(monotonic_model_var-theoretic_monotonic_model_var)/np.linalg.norm(theoretic_monotonic_model_var))

#%%

plt.figure(figsize=(plotwidth, plotwidth*1/2))
plt.title('Monotonic model mean')
plt.plot(X, monotonic_model_mean,color='r', label='Emperical Mean')
plt.plot(X, theoretic_monotonic_model_mean, color='b',label='Theoretic Mean')
plt.plot(X, monotonic_model[::50].T, color='g', alpha=0.05)
plt.plot(X, monotonic_model[0], color='g', alpha=0.05, label='Samples')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.savefig('figures/monotonic_mean.png', dpi=200,bbox_inches='tight')
plt.show()

plt.figure(figsize=(plotwidth, plotwidth*1/2))
plt.title('Monotonic model variance')
plt.plot(X, monotonic_model_var,color='r', label='Emperical Variance')
plt.plot(X, theoretic_monotonic_model_var, color='b',label='Theoretic Variance')
plt.xlabel('x')
plt.ylabel('Variance of f(x)')
plt.legend()
plt.savefig('figures/monotonic_variance.png', dpi=200,bbox_inches='tight')
plt.show()
#%%

m_mse = np.zeros(50)
v_mse= np.zeros(50)
m_rel = np.zeros(50)
v_rel= np.zeros(50)

n_grid = np.logspace(1, np.log10(N_samples), num=50, base=10, dtype=int)

for i,n  in enumerate(n_grid):
    m_mse[i] = np.mean((np.mean(monotonic_model[:n], axis=0)-theoretic_monotonic_model_mean)**2)
    v_mse[i] = np.mean((np.var(monotonic_model[:n], axis=0)-theoretic_monotonic_model_var)**2)

    m_rel[i] = np.linalg.norm(np.mean(monotonic_model[:n], axis=0)-theoretic_monotonic_model_mean)/np.linalg.norm(theoretic_monotonic_model_mean)
    v_rel[i] = np.linalg.norm(np.var(monotonic_model[:n], axis=0)-theoretic_monotonic_model_var)/np.linalg.norm(theoretic_monotonic_model_var)


#%%
plt.figure(figsize=(plotwidth, plotwidth*1/3))
plt.plot(n_grid, m_mse)

# plt.xscale('log')
# plt.yscale('log')
plt.xlabel('Number of samples')
plt.ylabel('MSE of empirical and theoretical Mean')
plt.title('Convergence of Mean')
plt.show()


# plt.xscale('log')
# plt.yscale('log')
plt.figure(figsize=(plotwidth, plotwidth*1/3))
plt.plot(n_grid, v_mse)
plt.xlabel('Number of samples')
plt.ylabel('MSE of empirical and theoretical Variance')
plt.title('Convergence of Variance')
plt.show()
# %%
# %%
plt.figure(figsize=(plotwidth, plotwidth*1/2))
plt.plot(n_grid, m_rel)

# plt.xscale('log')
# plt.yscale('log')
plt.xlabel('Number of samples')
plt.ylabel('relative error of mean')
plt.title('Convergence of Mean')
plt.savefig('figures/relative_convergence_of_monotonic_function_mean', dpi=200, bbox_inches='tight')
plt.show()

# plt.xscale('log')
# plt.yscale('log')
plt.figure(figsize=(plotwidth, plotwidth*1/2))
plt.plot(n_grid, v_rel)
plt.xlabel('Number of samples')
plt.ylabel('relative error of variance')
plt.title('Convergence of Variance')
plt.subplots_adjust(wspace=0.3)
plt.savefig('figures/relative_convergence_of_monotonic_function_var', dpi=200, bbox_inches='tight')
plt.show()

# %%
