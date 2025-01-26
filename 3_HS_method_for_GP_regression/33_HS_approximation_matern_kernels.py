#%%
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
from HSGP import HSGP_box_domain, se_spectral_density_1D, squared_exponential, matern_spectral_density_1D
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.size': 16.0,
    'axes.titlesize': 'medium'
})
textwidth = 5.9

# Prep parameters and data
kappa = 1
scale = 0.1
D = [-1, 1]
mlist = [12, 32, 64, 128]

tau = np.linspace(0, 5*scale, 100)
x = np.zeros_like(tau)
xprime = x + tau


# prepare plot
cmap = colormaps['tab10']
color_list = [cmap(i) for i in range(len(mlist)+1)]
lw_list = [4] + [1.2 + 0.3*i for i in range(len(mlist),0,-1)] 
alpha_list = [0.4] + [1]*(len(mlist))

plotscale = 0.95*2
plotwidth = plotscale*textwidth

plt.figure(figsize=(plotwidth,plotwidth*0.4))
plt.vlines(x=[3*i*(scale) for i in range(5)], linestyle= 'dashed', linewidth=1, color='grey', ymin=0,ymax=1)
plt.hlines(y=0, xmin=0, xmax=17*scale, linewidth=1, color='black')
plt.xticks([0, 5*scale],labels=[0,r'$5\cdot\ell$'])  # Add tick mark at 5*scale

# matern kernels
nus = [0.5, 1.5, 2.5, 3.5]

for i,nu in enumerate(nus):
    theta_K = [kappa, scale, nu]     # kappa, scale, nu
    y = theta_K[0]*Matern(length_scale=theta_K[1], nu=theta_K[2]).__call__(tau[:, None])[0]

    plt.plot(tau + (i*3)*scale, y, color=color_list[0], lw=lw_list[0], alpha=alpha_list[0])
    plt.text((i*3*scale) + 0.1, 0.95, r'$\nu=$'+str(nu))

    theta = [0] + theta_K

    # fit HSGP
    m = mlist[-1]
    GP = HSGP_box_domain(m, D, matern_spectral_density_1D, theta)
    
    
    Phix = GP.construct_phi(x[:, None])
    Phixprime = GP.construct_phi(xprime[:, None])
    Lambda = GP.construct_Lambda(GP.theta_k)

    for j, mval in enumerate(mlist):
        Ktilde = Phix[:,:mval] @ np.diag(Lambda[:mval]) @ Phixprime[:,:mval].T
        plt.plot(tau + (i*3)*scale, Ktilde[0,:],color=color_list[j+1], linewidth=lw_list[j+1])

# plt.legend()
# squared exponential kernel
i+=1
theta_K = [kappa, scale]
y = squared_exponential(tau*tau, *theta_K)


theta = [0] + theta_K
GP = HSGP_box_domain(m, D, se_spectral_density_1D, theta)


Phix = GP.construct_phi(x[:, None])
Phixprime = GP.construct_phi(xprime[:, None])
Lambda = GP.construct_Lambda(GP.theta_k)

plt.plot(tau + (i*3)*scale, y, color=color_list[0],lw=4, alpha=0.5)
plt.text((i*3*scale) + 0.1, 0.95, r'$\nu\to\infty$')
for j, mval in enumerate(mlist):
    Ktilde = Phix[:,:mval] @ np.diag(Lambda[:mval]) @ Phixprime[:,:mval].T
    plt.plot(tau + (i*3)*scale, Ktilde[0,:], color=color_list[j+1], linewidth=lw_list[j+1])

# fix legend...
custom_lines = [Line2D([0], [0], color=color_list[i], lw=lw_list[i], alpha=alpha_list[i]) for i in range(len(mlist)+1)]

plt.legend(custom_lines, ['true kernel']+[f'm={m}' for m in mlist], loc='center right')
plt.savefig('fig1.png', bbox_inches = 'tight')
plt.show()



# %%'m
