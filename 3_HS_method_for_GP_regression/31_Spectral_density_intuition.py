#%%
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 16.0,
    'axes.titlesize': 'medium'
})
textwidth = 5.9

# Define the squared exponential covariance function
def squared_exponential_covariance(r, length_scale=1.0):
    return np.exp(-0.5 * (r / length_scale) ** 2)

# Define the spectral density of the squared exponential covariance function
def spectral_density(frequency, length_scale=1.0):
    return np.sqrt(2 * np.pi) * length_scale * np.exp(-2 * (np.pi * frequency * length_scale) ** 2)

# Generate data for the plots
r = np.linspace(0, 10, 400)
l1 = 3
covariance = squared_exponential_covariance(r, length_scale=l1)

frequency = np.linspace(0, 10, 400)
density = spectral_density(frequency)

# Generate samples from the Gaussian process
num_samples = 10
mean = np.zeros_like(r)
cov_matrix = squared_exponential_covariance(np.abs(np.subtract.outer(r, r)), length_scale=l1)
samples = np.random.multivariate_normal(mean, cov_matrix, num_samples)

# Create the plots
scale = 0.95*2
plotwidth = scale*textwidth
fig, ax = plt.subplots(2, 3, figsize=(plotwidth,2/3*plotwidth))

# Plot the squared exponential covariance function
ax[0,0].plot(r, covariance, color='blue')
ax[0,0].set_title(f'Squared Exponential Covariance\n$\ell={l1}$ and $\kappa = 1$')
ax[0,0].set_xlabel('r')
ax[0,0].set_ylabel('Covariance')

# Plot the spectral density
ax[0,1].plot(frequency, density, color='orange')
ax[0,1].set_title(f'Spectral Density\n$\ell={l1}$ and $\kappa = 1$')
ax[0,1].set_xlabel('Frequency')
ax[0,1].set_ylabel('Density')
ax[0,1].set_xlim(0, 6)

# Plot the samples from the Gaussian process
for i in range(num_samples):
    ax[0,2].plot(r, samples[i], lw=0.5)
ax[0,2].set_title(f'Samples from Gaussian Process\n$\ell = {l1}$ and $\kappa=1$')
ax[0,2].set_xlabel('x')
ax[0,2].set_ylabel('f(x)')

# Generate data for the plots with length scale l2
l2 = 0.1
covariance_l2 = squared_exponential_covariance(r, length_scale=l2)
density_l2 = spectral_density(frequency, length_scale=l2)
cov_matrix_l2 = squared_exponential_covariance(np.abs(np.subtract.outer(r, r)), length_scale=l2)
samples_l2 = np.random.multivariate_normal(mean, cov_matrix_l2, num_samples)

# Plot the squared exponential covariance function with length scale l2
ax[1,0].plot(r, covariance_l2, color='blue')
ax[1,0].set_title(f'Squared Exponential Covariance\n$\ell={l2}$ and $\kappa = 1$',)
ax[1,0].set_xlabel('r')
ax[1,0].set_ylabel('Covariance')

# Plot the spectral density with length scale l2
ax[1,1].plot(frequency, density_l2, color='orange')
ax[1,1].set_title(f'Spectral Density\n$\ell={l2}$ and $\kappa = 1$')
ax[1,1].set_xlabel('Frequency')
ax[1,1].set_ylabel('Density')
ax[1,1].set_xlim(0, 6)

# Plot the samples from the Gaussian process with length scale l2
for i in range(num_samples):
    ax[1,2].plot(r, samples_l2[i], lw=0.5)
ax[1,2].set_title(f'Samples from Gaussian Process\n$\ell = {l2}$ and $\kappa=1$')
ax[1,2].set_xlabel('x')
ax[1,2].set_ylabel('f(x)')

# Show the plots
plt.tight_layout()
plt.savefig('spectral_density_intuition_plot.png', bbox_inches='tight')
plt.show()