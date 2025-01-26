#%%
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams.update({
    'font.size': 16.0,
    'axes.titlesize': 'medium'
})
textwidth = 5.9
plotwidth = 2*0.95*textwidth
plotheight = 0.35*plotwidth

models = ['baseline_GP', 'HS_model', 'VP_model']
# path = '6_Experiments/64_Ushaped_data/'
path = ''
model_names = ['baseline GP', 'HS model', 'VP model']

# fractions = np.arange(1,9)
x_samples = np.load(path + 'all_data/x_samples.npy' )
num_total = len(x_samples)
num_train = num_total - num_total//4

fractions = [0.5, 0.7272727272727273, 0.8888888888888888, 1.0, 2.0, 4.0]
num_train = ((np.array(fractions)*num_train)/8).astype(int)
print(num_train)

fig, ax = plt.subplots(1,2, figsize=(plotwidth,plotheight))
symbols = ['o', 'x', 's', '+']*2
R  = 7

for m, model in enumerate(models):
    rmses = np.zeros((len(fractions), R))
    lpds = np.zeros((len(fractions), R))

    for f,fraction in enumerate(fractions):
        for r in range(R):
            try:
                resultpath = path + f'Runs/Run_{r+1}/{model}/{fraction}/'
                rmse, lpd = np.load(resultpath + 'test_results.npy')
            except:
                resultpath = path + f'Runs/Run_{r+1}/{model}/{int(fraction)}/'
                rmse, lpd = np.load(resultpath + 'test_results.npy')

            rmses[f,r] = rmse
            lpds[f,r] = lpd

    rmse_mean = np.mean(rmses,axis=-1)
    rmse_std = np.std(rmses/np.sqrt(R),axis=-1)

    elpd_mean = np.mean(lpds,axis=-1)
    elpd_std = np.std(lpds/np.sqrt(R),axis=-1)

    ax[0].errorbar(num_train, rmse_mean, yerr=rmse_std, marker=symbols[m], linestyle='-', label=model_names[m], capsize=10, elinewidth=1, markeredgewidth=1.5)
    ax[1].errorbar(num_train, elpd_mean, yerr=elpd_std, marker=symbols[m], linestyle='-', label=model_names[m], capsize=10, elinewidth=1, markeredgewidth=1.5)

ax[0].set_xlabel('number of training points')
ax[0].set_ylabel('RMSE')
ax[0].set_xscale('log')
ax[0].set_xticks(num_train)
ax[0].set_xticklabels(num_train)

ax[1].set_xlabel('number of training points')
ax[1].set_ylabel('ELPD')
ax[1].set_xscale('log')
ax[1].set_xticks(num_train)
ax[1].set_xticklabels(num_train)

handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles=handles, labels=labels, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.1))

plt.tight_layout()
plt.savefig(path + 'performance_plot.png', bbox_inches='tight')
plt.show()

# # %%

# %%
