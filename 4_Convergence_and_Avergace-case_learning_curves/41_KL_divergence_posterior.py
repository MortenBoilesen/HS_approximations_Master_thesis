#%%
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from HSGP import HSGP_box_domain, GP_true, KL_divergence, squared_exponential, se_spectral_density_1D
import numpy as np
import matplotlib.pyplot as plt


# %%
kappa = 1
scale = 1
sigma = 0.1
theta = [sigma, kappa, scale]
d = 1
N = 10

GP_prior = GP_true(squared_exponential,theta)
GP_prior.fit(np.zeros((0, d)), np.zeros((0, 1)))

X_samples = np.random.uniform(-1,1,100)[: , None]
X_samples.sort(axis=0)

f_samples = GP_prior.generate_samples(X_samples, N)
y_samples = f_samples + np.random.normal(0, theta[0], f_samples.shape)


ms = [5, 10, 15, 20]
K = 100
X_eval = np.linspace(-1,1,100)[:, None]
domains = np.linspace(1.01, 10, K)
KL_divergence_means = np.zeros((4,K))
for j,m in enumerate(ms):
    KL_divergences = np.zeros((N,K))

    for k, d in enumerate(domains):
        D = np.array([-d,d])
        for i, yj in enumerate(y_samples):

            GP = HSGP_box_domain(m, D, se_spectral_density_1D, theta)
            GP.fit(X_samples, yj)
            mu_post, sigma_post = GP.predict(X_eval)


            GPt = GP_true(squared_exponential, theta)
            GPt.fit(X_samples, yj)
            mu_post_true, sigma_post_true = GPt.predict(X_eval)

            KL_divergences[i,k] = KL_divergence(mu_post, sigma_post, mu_post_true, sigma_post_true)


    KL_divergence_means[j] = np.mean(KL_divergences, axis=0)

#%%
    
plt.rcParams.update({
    'font.size': 11.0,
    'axes.titlesize': 'medium'
})
textwidth = 5.9
plotscale = 0.95
plotwidth = textwidth*plotscale

fig, ax = plt.subplots(1,figsize=(plotwidth,plotwidth*0.625))

for j,m in enumerate(ms):   
    ax.semilogy(np.linspace(1.1, 10, K), KL_divergence_means[j], label=f'm = {m}')

ax.set_xlabel("L")
ax.set_ylabel("log KL divergence")
ax.set_title("KL divergence between approximate and true posterior")
ax.legend()

ax.set_xticks(np.arange(1, 10), minor=True)
ax.set_yticks([], minor=True)
plt.savefig('fig4_convergence.png', bbox_inches='tight')
plt.show()
# %%
