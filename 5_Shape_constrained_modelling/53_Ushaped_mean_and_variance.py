#%%
import os, sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
# 
from HSGP import GP_true, squared_exponential
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 16.0,
    'axes.titlesize': 'medium'
})
textwidth = 5.9
plotscale = 0.95
plotwidth = textwidth*plotscale

from scipy.special import erf

def omega(x,a,scale,eta):
    xma = x-a
    sqrt2 = np.sqrt(2)
    sqrtpi = np.sqrt(np.pi)
    sqrt2cube = 2**(1.5)
    scalesq=scale*scale
    # etasq = eta*eta
#

    u = (xma)/sqrt2/scale

    erf_term = (2*xma**3/(3*scale) + 2*xma*(1-scale))*erf(u)
    exp_term = ((2*sqrtpi-5)*scalesq*sqrt2 + sqrt2cube*xma*xma)/(3*sqrtpi)*np.exp(-(u*u))
    const_term = ((5-2*sqrtpi)*sqrt2*scalesq/3 - xma*xma*sqrt2)/sqrtpi

    return eta*eta*scalesq*sqrtpi/sqrt2*(erf_term + exp_term + const_term)


def theoretical_var(X,theta,L):
    _, eta, scale = theta
    a = -L
    rho = scale/np.sqrt(2)
    return 2*eta*eta*omega(x=X,a=a,scale=rho,eta=eta)


#%%

#sample from GP
# theta = [0.1, 2, 0.8]
theta = [0.1, 1.5, 1.1]

N_samples = 10000
N_x = 1000
L = 5
f0 = -2*L*theta[1]**2/2
F0 = 0
X = np.linspace(-L,L,N_x)

try: 
    u_shaped_model = np.load(f'ushaped_theta={theta}.npy'.replace(', ','-'))
    
except:
    print('Generating data')
    GP = GP_true(squared_exponential,theta)
    GP.fit(np.zeros((0, 1)), np.zeros((0, 1)))

    dx = 2*L/N_x
    samples = GP.generate_samples(X[:, None], N_samples).reshape(N_samples,N_x)


    # model the u-shaped function
    monotonic_model = np.zeros(samples.shape)
    u_shaped_model = np.zeros(samples.shape)

    positive_model = samples**2
    monotonic_model[:,0] = f0
    u_shaped_model[:,0] = F0
    for i in range(1,N_x):
        monotonic_model[:,i] = f0 + np.trapz(positive_model[:,:i], dx=dx, axis=1)
        u_shaped_model[:,i] = F0 + np.trapz(monotonic_model[:,:i], dx=dx, axis=1)

    np.save(f'ushaped_theta={theta}'.replace(', ','-'),u_shaped_model)

#%%

#calculate the mean and variance of the monotonic model and the theoretical values
u_shaped_model_mean = np.mean(u_shaped_model, axis=0)
u_shaped_model_var = np.var(u_shaped_model, axis=0)

z = (X+L)/theta[2]
theoretic_u_shaped_model_mean = F0 + f0*(X+L) + theta[1]**2/2*(X+L)**2
theoretic_u_shaped_model_var = theoretical_var(X,theta,L)

print('MSE between theoretic and emperical mean:', np.mean((u_shaped_model_mean-theoretic_u_shaped_model_mean)**2))
print('MSE between theoretic and emperical variance:', np.mean((u_shaped_model_var-theoretic_u_shaped_model_var)**2))

print('Relative error between theoretic and emperical mean:', np.linalg.norm(u_shaped_model_mean-theoretic_u_shaped_model_mean)/np.linalg.norm(theoretic_u_shaped_model_mean))
print('Relative error between theoretic and emperical variance:', np.linalg.norm(u_shaped_model_var-theoretic_u_shaped_model_var)/np.linalg.norm(theoretic_u_shaped_model_var))

#%%
plt.figure(figsize=(plotwidth, plotwidth*1/2))
plt.title('U-shaped model mean')
plt.plot(X, u_shaped_model_mean,color='r', label='Emperical Mean')
plt.plot(X, theoretic_u_shaped_model_mean, color='b',label='Theoretic Mean')
plt.plot(X, u_shaped_model[::50].T, color='g', alpha=0.05)
plt.plot(X, u_shaped_model[0], color='g', alpha=0.05, label='Samples')
plt.legend()
plt.xlabel('x')
plt.ylabel('F(x)')
plt.savefig('figures/ushaped_mean.png', dpi=200, bbox_inches='tight')
plt.show()

plt.figure(figsize=(plotwidth, plotwidth*1/2))
plt.title('U-shaped model variance')
plt.plot(X, u_shaped_model_var,color='r', label='Emperical Variance')
plt.plot(X, theoretic_u_shaped_model_var, color='b',label='Theoretic Variance')
plt.legend()
plt.xlabel('x')
plt.ylabel('Variance of F(x)')
plt.savefig('figures/ushaped_variance.png', dpi=200,bbox_inches='tight')
plt.show()


#%%

m_mse = np.zeros(50)
v_mse= np.zeros(50)
m_rel = np.zeros(50)
v_rel= np.zeros(50)

n_grid = np.logspace(1, 4, num=50, base=10, dtype=int)

for i,n  in enumerate(n_grid):
    m_mse[i] = np.mean((np.mean(u_shaped_model[:n], axis=0)-theoretic_u_shaped_model_mean)**2)
    v_mse[i] = np.mean((np.var(u_shaped_model[:n], axis=0)-theoretic_u_shaped_model_var)**2)

    m_rel[i] = np.linalg.norm(np.mean(u_shaped_model[:n], axis=0)-theoretic_u_shaped_model_mean)/np.linalg.norm(theoretic_u_shaped_model_mean)
    v_rel[i] = np.linalg.norm(np.var(u_shaped_model[:n], axis=0)-theoretic_u_shaped_model_var)/np.linalg.norm(theoretic_u_shaped_model_var)



#%%
plt.figure(figsize=(plotwidth, plotwidth*1/2))
plt.plot(n_grid, m_mse)
plt.xlabel('Number of samples')
plt.ylabel('MSE of empirical and theoretical Mean')
plt.title('Convergence of Mean')
plt.show()

plt.figure(figsize=(plotwidth, plotwidth*1/2))
plt.plot(n_grid, v_mse)
plt.xlabel('Number of samples')
plt.ylabel('MSE of empirical and theoretical Variance')
plt.title('Convergence of Variance')
plt.show()
# %%
plt.figure(figsize=(plotwidth, plotwidth*1/2))
plt.plot(n_grid, m_rel)
plt.xlabel('Number of samples')
plt.ylabel('relative error of mean')
plt.title('Convergence of Mean')
plt.savefig('figures/relative_convergence_of_ushaped_function_mean', dpi=200, bbox_inches='tight')
plt.show()

plt.figure(figsize=(plotwidth, plotwidth*1/2))
plt.plot(n_grid, v_rel)
plt.xlabel('Number of samples')
plt.ylabel('relative error of variance')
plt.title('Convergence of Variance')
plt.savefig('figures/relative_convergence_of_ushaped_function_var', dpi=200, bbox_inches='tight')
plt.show()

# %%
