#%%
import sys
import os

# Add parent directory to the sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# path = 'Generalization_error/'
path = ''

plt.rcParams.update({
    'font.size': 14.0,
    'axes.titlesize': 'medium'
})
textwidth = 5.9
plotwidth = 2*0.95*textwidth
plotheight = plotwidth*0.7


D = 100
S = 1000
R = 40
sigma = 0.1
sigmax = 1
L = 10

# theta = [sigma, kappa, scale]

n_step = 10
N = np.arange(5,S,n_step)

# def lambda_approx(k, scale, L):
#     sqrt_lambda =  k*np.pi/(2*L)
#     l = sqrt_lambda*sqrt_lambda
#     return np.sqrt(2*np.pi)*scale*np.exp(-0.5*scale**2*l)


def lower_bound_true(n, sigma, scale, sigma_x, k_max=50000):
    """
    Compute the lower bound on the generalization error for a given number of training points.

    Args:
        n (int): Number of training points.
        sigma (float): Standard deviation of the noise.
        scale (float): Scaling factor.
        sigma_x (float): Standard deviation of the input data.
        k_max (int, optional): Maximum number of terms in the sum. Defaults to 50000.

    Returns:
        float: The lower bound on the generalization error.
    """

    a = 1/(4*sigma_x**2)
    b = 1/(2*scale**2)
    c = np.sqrt(a**2 + 2*a*b)
    A = a+b+c
    B = b/A

    l = np.sqrt(2*a/A)*B**np.arange(1,k_max+1)

    return sigma**2*np.sum(l/(n*l + sigma**2))

# def lower_bound_approx(n, sigma, scale, k_max=50000):
#     """
#     Compute the lower bound on the generalization error for a given number of training points.

#     Args:
#         n (int): Number of training points.
#         sigma (float): Standard deviation of the Gaussian process.
#         scale (float): Scaling factor.
#         k_max (int, optional): Maximum number of terms in the sum. Defaults to 50000.

#     Returns:
#         float: The lower bound on the generalization error.
#     """
#     sqrt_lambda =  np.arange(1,k_max+1)*np.pi/(2*L)
#     l = np.sqrt(2*np.pi)*scale*np.exp(-0.5*scale**2*sqrt_lambda*sqrt_lambda)

#     return sigma**2*np.sum(l/(n*l + sigma**2))


scales = [0.05, 0.1, 1, 5]
scalenames = ["005", "01","10", "50"]
mlist = [12, 32, 64, 128, 256]

fig, ax = plt.subplots(2,2, figsize=(plotwidth, plotheight))
for i, scale in enumerate(scales):
    j = i // 2
    k = i % 2
    generalizarion_error = np.load(path + f'generalization_error/learning_curve_full_scale={scalenames[i]}_D={D}_nstep={n_step}.npy')
    learning_curves = generalizarion_error.mean(-1)
    
    mean_curve = learning_curves.mean(-1)
    ax[j,k].plot(N, mean_curve, label=f'full model', linestyle='-.')

    std_curve  = learning_curves.std(-1)
    lower = mean_curve - std_curve
    upper = mean_curve + std_curve 
    ax[j,k].fill_between(N, lower, upper, alpha=0.3)

    for m in mlist:
        generalizarion_error = np.load(path + f'generalization_error/learning_curve_HSGP_scale={scalenames[i]}_D={D}_R={R}_nstep={n_step}_m={m}.npy')
        learning_curves = generalizarion_error.mean(-1)
        
        mean_curve = learning_curves.mean(-1)
        ax[j,k].plot(N, mean_curve, label=f'm = {m}', linestyle='--')

        std_curve  = learning_curves.std(-1)
        lower = mean_curve - 2*std_curve
        upper = mean_curve + 2*std_curve 
        ax[j,k].fill_between(N, lower, upper, alpha=0.3)

        # upper = np.percentile(learning_curves, 5, axis=1)
        # lower = np.percentile(learning_curves, 95, axis=1)

    params = [sigma, sigmax, scale]
    ax[j,k].plot(N, [lower_bound_true(n, sigma, scale, sigmax) for n in N], label='Lower Bound')
    # ax[j,k].plot(N, [lower_bound_approx(n, sigma, scale ) for n in N], label='HS Lower Bound', color='lightgreen')
    ax[j,k].set_title(f'scale={scale}')
    ax[j,k].set_xlabel('n')
    ax[j,k].set_yscale('log')

    if ax[j,k].get_yscale == 'log':
        ax[j,k].set_ylabel('log learning curve')
    else:
        ax[j,k].set_ylabel('learning curve')

    # Update the legend with the new handles


# Add a new entry to the legend
grey_patch = Patch(color='grey', alpha=0.2, label='95% CI')

# Get existing handles and labels
handles, labels = ax[j,k].get_legend_handles_labels()

# Add the grey patch to the handles
handles.append(grey_patch)
fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5,-0.07))


plt.suptitle('Learning curves for different scale parameters')
plt.tight_layout()
plt.savefig('Generalization_error.png', dpi=500, bbox_inches='tight')
plt.show()


# %%
