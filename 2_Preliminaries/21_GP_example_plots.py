#%%

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..") 
from HSGP import GP_true, squared_exponential, HSGP_box_domain, se_spectral_density_1D, se_spectral_density_grad

#%%
### PLOT ILLUSTRATING THE SQUARED EXPONENTIAL KERNEL FUNCTION
X_samples = np.linspace(-1,1,100)[: , None]
kappa = 1
sigma = 0.1
d = 1
N = 25

scales = [0.05, 0.5, 5]
fig, axes = plt.subplots(2, 3, figsize=(18,12))

for i, scale in enumerate(scales):
    theta = [sigma, kappa, scale]
    GP_prior = GP_true(squared_exponential, theta)
    GP_prior.fit(np.zeros((0, d)), np.zeros((0, 1)))
    
    f_samples = GP_prior.generate_samples(X_samples, N)
    y_samples = f_samples + np.random.normal(0, theta[0], f_samples.shape)
    
    axes[1, i].plot(X_samples, f_samples[0], label=f'Samples')
    
    for j in range(1, N):
        axes[1, i].plot(X_samples, f_samples[j])

    axes[1, i].set_title(f'Samples from GP with (scale={scale})')
    axes[1, i].set_xlabel('x')
    axes[1, i].set_ylabel('f(x)')
    
    pairwise_diff = np.abs(X_samples - X_samples.T)
    cov_matrix = squared_exponential(pairwise_diff, kappa, scale)
    im = axes[0, i].imshow(cov_matrix, cmap='hot', interpolation='nearest', extent=[X_samples.min(), X_samples.max(), X_samples.min(), X_samples.max()])
    axes[0, i].set_title(f'Squared exponential function w. (scale={scale})')
    axes[0, i].set_xlabel('x')
    cbar = fig.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)
    axes[0, i].set_ylabel('x\'')

plt.show()

# %%
### PLOT ILLUSTRATING THE PRIOR AND POSTERIOR DISTRIBUTION.
np.random.seed(1)
N = 10
theta_init = [1, 1, 1]
x_train = np.random.uniform(-5, 5, 5)
y_train = np.sin(x_train) + np.random.normal(0, 0.1, x_train.shape)
x_pred = np.linspace(-10, 10, 100)

GP = GP_true(squared_exponential, theta=theta_init)
theta = GP.optimize_hyperparameters(X=x_train[:,None], y=y_train[:,None],theta_init=theta_init)
GP.fit(X=x_train[:,None], y=y_train[:,None])
post_mu, post_var = GP.predict(x_pred[:,None])

# Test
f_samples = GP.generate_samples(x_pred[:,None], num_samples=N)

GP_prior = GP_true(squared_exponential, theta=theta)
GP_prior.fit(np.zeros((0, d)), np.zeros((0, 1)))
prior_mean, prior_var = GP_prior.predict(x_pred[:,None])
f_prior_samples = GP_prior.generate_samples(x_pred[:,None], N)
plt.rcParams.update({'font.size': 14})
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# Plot the prior mean and variance

axes[0].plot(x_train, y_train, 'ro', label='Training Data')
axes[0].plot(x_pred, prior_mean[:,0], 'b-', label='Prior Mean')
axes[0].fill_between(x_pred, prior_mean[:,0] - 2*np.sqrt(np.diag(prior_var)), prior_mean[:,0] + 2*np.sqrt(np.diag(prior_var)), color='blue', alpha=0.2, label='95% Confidence Interval')
for i in range(f_prior_samples.shape[0]):
    axes[0].plot(x_pred, f_prior_samples[i], 'g-', alpha=0.5, label=f'$f$ samples' if i == 0 else "")
axes[0].set_title('Prior distribution of $f$')
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].legend()

# Plot the training data, posterior mean, and variance
axes[1].plot(x_train, y_train, 'ro', label='Training Data')
axes[1].plot(x_pred, post_mu[:,0], 'b-', label='Posterior Mean')
axes[1].fill_between(x_pred, post_mu[:,0] - 2*np.sqrt(np.diag(post_var)), post_mu[:,0] + 2*np.sqrt(np.diag(post_var)), color='blue', alpha=0.2, label='95% Confidence Interval')
for i in range(f_samples.shape[0]):
    axes[1].plot(x_pred, f_samples[i,:], 'g-', alpha=0.5, label=f'$f$ samples' if i == 0 else "")
axes[1].set_title('Posterior predective distribution of $f$')
axes[1].set_xlabel('x')
axes[1].set_ylabel('f(x)')
axes[1].legend()

plt.tight_layout()
plt.show()

