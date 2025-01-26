#%%
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..") 
from HSGP import GP_true, squared_exponential, HSGP_box_domain, se_spectral_density_1D, se_spectral_density_grad

d = 1

# PLOT ILLUSTRATING  THE HS APPROXIMATION

np.random.seed(4)
N = 10
theta_init = [1, 1, 1]
x_train = np.random.uniform(-5, 5, 5)
y_train = np.sin(x_train) + np.random.normal(0, 0.1, x_train.shape)
x_pred = np.linspace(-12, 12, 100)

ms = [2, 4, 8, 16]
N = 10
D = [-10,10]

GP = GP_true(squared_exponential, theta=theta_init)
theta = GP.optimize_hyperparameters(X=x_train[:,None], y=y_train[:,None],theta_init=theta_init)
GP.fit(X=x_train[:,None], y=y_train[:,None])
post_mu, post_var = GP.predict(x_pred[:,None])
print(theta)
# Test


f_samples = GP.generate_samples(x_pred[:,None], num_samples=N)

GP_prior = GP_true(squared_exponential, theta=theta)
GP_prior.fit(np.zeros((0, d)), np.zeros((0, 1)))
prior_mean, prior_var = GP_prior.predict(x_pred[:,None])
f_prior_samples = GP_prior.generate_samples(x_pred[:,None], N)

fig, axes = plt.subplots(len(ms)+1, 2, figsize=(20, 20))
plt.rcParams.update({'font.size': 14})

# Plot the prior mean and variance

axes[0,0].plot(x_train, y_train, 'ro', label='Training Data')
axes[0,0].plot(x_pred, prior_mean[:,0], 'b-', label='Prior Mean')
axes[0,0].fill_between(x_pred, prior_mean[:,0] - 2*np.sqrt(np.diag(prior_var)), prior_mean[:,0] + 2*np.sqrt(np.diag(prior_var)), color='blue', alpha=0.2, label='95% Confidence Interval')
for i in range(f_prior_samples.shape[0]):
    axes[0,0].plot(x_pred, f_prior_samples[i,:], 'g-', alpha=0.5, label=f'$f$ samples' if i == 0 else "")
axes[0,0].set_title('Prior distribution of full Gaussian process $f$')
axes[0,0].set_xlabel('x')
axes[0,0].set_ylabel('f(x)')

# Plot the training data, posterior mean, and variance
axes[0,1].plot(x_train, y_train, 'ro', label='Training Data')
axes[0,1].plot(x_pred, post_mu[:,0], 'b-', label='Posterior Mean')
axes[0,1].fill_between(x_pred, post_mu[:,0] - 2*np.sqrt(np.diag(post_var)), post_mu[:,0] + 2*np.sqrt(np.diag(post_var)), color='blue', alpha=0.2, label='95% Confidence Interval')
for i in range(f_samples.shape[0]):
    axes[0,1].plot(x_pred, f_samples[i,:], 'g-', alpha=0.5, label=f'$f$ samples' if i == 0 else "")
axes[0,1].set_title('Posterior predective distribution of full Gaussian process $f$')
axes[0,1].set_xlabel('x')

#Comparisson between HS model and true GP

for k, m in enumerate(ms):

    GP_approx = HSGP_box_domain(m,D, se_spectral_density_1D, theta=theta, spectral_density_grad=se_spectral_density_grad)
    # theta = GP_approx.optimize_hyperparameters(X=x_train[:,None], y=y_train[:,None],theta_init=theta_init)
    print(theta)
    GP_approx.fit(X=x_train[:,None], y=y_train[:,None])
    post_mu_approx, post_var_approx = GP_approx.predict(x_pred[:,None])
    # Test
    f_approx_samples = GP_approx.generate_samples(x_pred[:,None], num_samples=N)

    GP_approx_prior = HSGP_box_domain(m,D, se_spectral_density_1D, theta=theta_init)
    GP_approx_prior.fit(np.zeros((0, d)), np.zeros((0, 1)))
    prior_mean_approx, prior_var_approx = GP_approx_prior.predict(x_pred[:,None])
    f_approx_prior_samples = GP_approx_prior.generate_samples(x_pred[:,None], N)

    # Plot the prior mean and variance

    axes[k + 1,0].plot(x_train, y_train, 'ro', label='Training Data')
    axes[k + 1,0].plot(x_pred, prior_mean_approx[:,0], 'b-', label='Prior Mean')
    axes[k + 1,0].fill_between(x_pred, prior_mean_approx[:,0] - 2*np.sqrt(np.diag(prior_var_approx)), prior_mean_approx[:,0] + 2*np.sqrt(np.diag(prior_var_approx)), color='blue', alpha=0.2, label='95% Confidence Interval')
    for i in range(f_approx_prior_samples.shape[1]):
        axes[k + 1,0].plot(x_pred, f_approx_prior_samples[:,i], 'g-', alpha=0.5, label=f'$f$ samples' if i == 0 else "")
    axes[k + 1,0].set_title(f'HS approximation $f$ with {m} basis functions')
    axes[k + 1,0].set_xlabel('x')
    axes[k + 1,0].set_ylabel('f(x)')

    # Plot the training data, posterior mean, and variance
    axes[k + 1,1].plot(x_train, y_train, 'ro', label='Training Data')
    axes[k + 1,1].plot(x_pred, post_mu_approx[:,0], 'b-', label='Posterior Mean')
    axes[k + 1,1].fill_between(x_pred, post_mu_approx[:,0] - 2*np.sqrt(np.diag(post_var_approx)), post_mu_approx[:,0] + 2*np.sqrt(np.diag(post_var_approx)), color='blue', alpha=0.2, label='95% Confidence Interval')
    for i in range(f_approx_samples.shape[1]):
        axes[k + 1,1].plot(x_pred, f_approx_samples[:,i], 'g-', alpha=0.5, label=f'$f$ samples' if i == 0 else "")
    axes[k + 1,1].set_title(f'HS approximation $f$ with {m} basis functions')
    axes[k + 1,1].set_xlabel('x')
    axes[k + 1,1].set_ylabel('f(x)')
    # Add legends to all subplots
    handles, labels = [], []
    for ax in axes.flatten():
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc='lower center', ncol=4)

plt.tight_layout()
plt.show()

# %%
