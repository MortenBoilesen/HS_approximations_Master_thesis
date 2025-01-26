#%%
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# path = '6_Experiments/63_Ushaped_benchmark_functions/HS_model/'
path = ''
b_function = 'sine'    

num_chains = 3
num_samples = 1000
num_experments = 3

p_path = path + 'results/' + b_function + '/' + str(num_samples) + '/parameters_and_rhats/' 


## EXTRACT F SAMPLES AND PLOT ??

# m=10, L=30
# m=2, l=30 !!!
m = 10
L = 30

params = np.load(p_path +f'parameter_arrays.npy')
rhats = np.load(p_path +f'rhats.npy')


# params = np.load(p_path +f'parameters_m={m}_L={L}.npy')
# rhats = np.load(p_path +f'rhats_m={m}_L={L}.npy')
r_test = rhats <= 1.1
chain_test = np.prod(r_test, axis=-1, dtype=bool)
run_test = np.prod(chain_test, axis = -1, dtype=bool)

for r in range(num_experments):
    print(r_test[r])
    print(chain_test[r])
    print()


#%%
f0 = params[:,0]
F0 = params[:,1]
f0_param = params[:,2]
F0_param = params[:,3]
sigma = params[:,4]
kappa = params[:,5]
scale = params[:,6]
alpha = params[:,7:]



for r in range(num_experments):
    if run_test[r]:
        continue
    fig, ax = plt.subplots(3,num_chains, figsize=(6*num_chains,12))

    for c in range(num_chains):
        if len(params[r]) <= 10:
            colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
        else:
            colors = list(plt.cm.tab20.colors)
            colors += colors

        idx = np.arange(c,num_samples*num_chains,num_chains)
        ax[0,c].plot(kappa[r,idx], alpha = 0.9, label= f"kappa ({rhats[r,c,5]:.3})", color = colors.pop(0))
        ax[0,c].plot(scale[r,idx], alpha = 0.9, label = f"scale ({rhats[r,c,4]:.3})", color = colors.pop(0))
        ax[0,c].plot(sigma[r,idx], alpha = 0.9, label = f"sigma ({rhats[r,c,3]:.3})", color = colors.pop(0))
        ax[0,c].set_title(f'Chain {c}, converged {chain_test[r,c]}')


        ax[1,c].plot(f0[r,idx], alpha=0.9, label=f"f0 ({rhats[r,c,0]:.3})", color = colors.pop(0))
        ax[1,c].plot(F0[r,idx], alpha=0.9, label=f"F0 ({rhats[r,c,1]:.3})")
        ax[1,c].plot(f0_param[r,idx], alpha=0.9, label=f"f0_param ({rhats[r,c,2]:.3})")
        ax[1,c].plot(F0_param[r,idx], alpha=0.9, label=f"F0_param ({rhats[r,c,3]:.3})")

        # Plotting alphas separately
        for i in range(len(alpha[r])):
            if rhats[r,c, 6+i] >= 1.1 or i <= 5:
                ax[2,c].plot(alpha[r,i][idx], alpha=0.9, label=f"alpha_{i} ({rhats[r,c, 6+i]:.3})", color = colors.pop(0))


        for i in range(3):
            ax[i,c].legend()
            ax[i,c].set_ylabel('parameter')
            ax[i,c].set_xlabel('n')
        
    plt.suptitle(f'Run: {r} ({run_test[r]})', fontsize=16)
    plt.tight_layout()
    plt.show()

# %%
