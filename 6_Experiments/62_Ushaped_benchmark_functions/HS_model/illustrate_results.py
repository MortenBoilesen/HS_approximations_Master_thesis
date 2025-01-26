#%%
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils import benchmark_ushaped, compute_rhat

# path = '6_Experiments/63_Ushaped_benchmark_functions/HS_model/'
path = ''

benchmark_function_names = ['flat', 'skew', 'parabola', 'abs', 'sine', 'step']
    

# x_test = np.load(datapath+'x_test.npy')
# y_test = np.load(datapath+'y_test.npy')
# x_pred = np.linspace( -1.8, 1.8 , 100 )


## EXTRACT F SAMPLES AND PLOT ??

fig, ax = plt.subplots(2,3, figsize=(10,8))

xpred = np.linspace(-5,5, 100)


for i in range(6):
    b_function = benchmark_function_names[i]
    row = i // 3
    col = i % 3
    ftrue = benchmark_ushaped(i, xpred)

    xtrain = np.load(path +'results/' + b_function + '/best_model_xtrain.npy')
    ytrain = np.load(path +'results/' + b_function + '/best_model_ytrain.npy')
    fpred = np.load(path +'results/' + b_function + '/best_model_fpred.npy')


    ax[row, col].plot(xpred, ftrue, label="True f")
    ax[row, col].scatter(xtrain, ytrain, label="Training data", color = "tab:orange")
    ax[row, col].plot(xpred, fpred[:,0], alpha=0.1, color="tab:green", label="samples")
    ax[row, col].plot(xpred, fpred[:,1:100], alpha=0.2, color="tab:green")
    ax[row, col].set_title(b_function)

plt.legend()
plt.tight_layout()
plt.show()

# %%
num_chains = 3
num_samples = 1000
params = np.load(path+'results/' + b_function + '/best_model_param_samples.npy')
rs = compute_rhat(parameter_array=params, num_chains=num_chains, num_samples=num_samples)
r_test = rs <= 1.1
print(r_test)


f0 = params[0]
F0 = params[1]
f0_param = params[2]
F0_param = params[3]
sigma = params[4]
kappa = params[5]
scale = params[6]
alpha = params[7:]

fig, ax = plt.subplots(3,num_chains, figsize=(6*num_chains,12))



for c in range(num_chains):
    if len(params) <= 10:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    else:
        colors = list(plt.cm.tab20.colors)

    idx = np.arange(c,num_samples*num_chains,num_chains)
    ax[0,c].plot(kappa[idx], alpha = 0.9, label= f"kappa ({r_test[c,5]})", color = colors.pop(0))
    ax[0,c].plot(scale[idx], alpha = 0.9, label = f"scale ({r_test[c,4]})", color = colors.pop(0))
    ax[0,c].plot(sigma[idx], alpha = 0.9, label = f"sigma ({r_test[c,3]})", color = colors.pop(0))

    ax[1,c].plot(f0[idx], alpha=0.9, label=f"f0 ({r_test[c,0]})", color = colors.pop(0))
    ax[1,c].plot(F0[idx], alpha=0.9, label=f"F0 ({r_test[c,1]})")
    ax[1,c].plot(f0_param[idx], alpha=0.9, label=f"f0_param ({r_test[c,2]})")
    ax[1,c].plot(F0_param[idx], alpha=0.9, label=f"F0_param ({r_test[c,3]})")

    # Plotting alphas separately
    for i in range(len(alpha)):
        ax[2,c].plot(alpha[i][idx], alpha=0.9, label=f"alpha_{i} ({r_test[c, 6+i]})", color = colors.pop(0))

    ax[0,c].legend()
    ax[1,c].legend()
    ax[2,c].legend()


plt.ylabel('parameter')
plt.xlabel('n')
plt.show()

# %%
