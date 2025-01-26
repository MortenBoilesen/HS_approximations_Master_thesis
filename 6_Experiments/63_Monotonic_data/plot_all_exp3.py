#%% BASELINE MODEL
import numpy as np
import pickle
from matplotlib import pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import plot_gp_with_samples, plot_with_samples
plt.rcParams.update({'font.size': 12})  # Set the fontsize to 12


models = ['baseline_GP', 'HS_model', 'VP_model', 'SDE_model']



# path = '5_Shape_constrained_modelling/53_Relaxing_convexity/'
path = ''
datapath = path + '00data/Fertility_rate/'

# Load data
x_test = np.load(datapath+'x_test.npy')
y_test = np.load(datapath+'y_test.npy')
x_train = np.load(datapath+'x_train.npy')
y_train = np.load(datapath+'y_train.npy')

x_pred = np.linspace( np.min(x_train), np.max(x_test), 100)

fig, ax = plt.subplots(1,len(models), figsize=(16,5))


for m, model in enumerate(models):
    resultpath = path+model+'/results/'
    results = np.load(resultpath+'test_results.npy')
    fpred = np.load(resultpath+'test_fpred.npy')

    if model == 'baseline_GP': # or model == 'SDE_model':
        mu = np.load(resultpath+'test_mu.npy')
        std = np.load(resultpath+'test_std.npy')
        plot_gp_with_samples(ax[m], x_pred, fpred, mu, std, alpha_samples=0.1, num_samples=100)
        model_selection_results = {}
    else:
        fpred = np.load(resultpath+'test_fpred.npy')
        sigma = np.load(resultpath+ 'test_sigma.npy')
        plot_with_samples(ax[m], x_pred, fpred, sigma, alpha_samples=0.1, num_samples=100)

        with open(resultpath+'model_selection_results.pkl', 'rb') as f:
            model_selection_results = pickle.load(f)

    lpd, _ = np.load(resultpath + 'test_results.npy')

    ax[m].scatter(x_train,y_train,label="Training data", color="tab:blue")
    ax[m].scatter(x_test,y_test,label="Testing data", color="tab:orange")
    ax[m].set_xlabel('Standardized year')
    ax[m].set_ylabel('Standardized fertility rate')

    title = model.replace('_',' ')
    title += ',\n'
    for key, val, in model_selection_results.items():
        title += key + ' : ' + str(val) + '   '
    ax[m].set_title(title)

    # ax[m].set_title(f'model = {model} with lpd = {np.mean(lpd):.3}')


    # if model == 'VP_model':
    #     x_virtual = np.load(path+model+'/results/x_virtual.npy')
    #     lower, upper = ax[m].get_ylim()
    #     ax[m].scatter(x_virtual, -1*np.ones(x_virtual.shape)+0.05, marker='^', label="Virtual points")

    # ax[m].set_ylim(-1,2)

        
# fig.subplots_adjust(left=0.22, right=0.95)
ax[0].legend(loc = 'lower left')
# ax[-1].legend(loc='center right',bbox_to_anchor=(1.6,0.5), fancybox=True, shadow=True)  # Add legend on the left side

fig.tight_layout()
plt.savefig(path+'plot_ex5.png')
plt.show()



# %%
