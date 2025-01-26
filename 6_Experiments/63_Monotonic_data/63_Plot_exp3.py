#%% BASELINE MODEL
import numpy as np
from matplotlib import pyplot as plt
from utils import plot_with_samples, plot_gp_with_samples
import pickle

plt.rcParams.update({
    'font.size': 16.0,
    'axes.titlesize': 'medium'
})
textwidth = 5.9
plotwidth = 2*0.95*textwidth
plotheight = plotwidth*0.6


models = ['baseline_GP', 'HS_model', 'VP_model', 'SDE_model']

datapath = '5_Shape_constrained_modelling/53_Relaxing_convexity/00data/Fertility_rate/'
path = '5_Shape_constrained_modelling/53_Relaxing_convexity/'

year_std = 18.184242262647807
year_mean = 1991

y_mean = 3.4063193866029415
y_std = 0.9264302087414242

# Load data
renormalized = True

x_train = np.load(datapath+'x_train.npy')
x_test = np.load(datapath+'x_test.npy')
y_test = np.load(datapath+'y_test.npy')
y_train = np.load(datapath+'y_train.npy')

if renormalized:
    x_train = x_train*year_std + year_mean
    x_test = x_test*year_std + year_mean
    y_train = y_train*y_std + y_mean
    y_test = y_test*y_std + y_mean

x_pred = np.linspace( np.min(x_train), np.max(x_test), 100)

fig, ax = plt.subplots(2, 2, figsize=(plotwidth,plotheight))

for j, model in enumerate(models):
    i = j//2
    k = j % 2
    try:
        results = np.load(path+model+f'/results/10000_test_results.npy')        
        fpred = np.load(path+model+f'/results/10000_test_fpred.npy')
        sigma = np.load(path+model+f'/results/10000_test_sigma.npy')
    except:
        results = np.load(path+model+f'/results/test_results.npy')        
        fpred = np.load(path+model+f'/results/test_fpred.npy')
        sigma = np.load(path+model+f'/results/test_sigma.npy')

    if renormalized:
        fpred = fpred*y_std + y_mean
        sigma = sigma*y_std
    plot_with_samples(ax[i,k], x_pred, fpred, sigma, num_samples = 100)
    
    ax[i,k].scatter(x_test,y_test,label="Testing data", color="tab:orange")
    ax[i,k].scatter(x_train,y_train,label="Training data", color="tab:blue")
    ax[i,k].set_ylabel('Fertility rate')


    if model == 'VP_model':
        with open(path+model+f'/results/model_selection_results.pkl', 'rb') as file:
            best_config = pickle.load(file)

        num_virtual_points = best_config['num_virtual']
        x_virtual = np.linspace(np.min(x_test), np.max(x_test), num_virtual_points)
        lower, upper = ax[i,k].get_ylim()
        if renormalized:
            ax[i,k].scatter(x_virtual, 0*np.ones(x_virtual.shape)+0.05, marker='^', label="Virtual points")
        else:
            ax[i,k].scatter(x_virtual, -3*np.ones(x_virtual.shape)+0.05, marker='^', label="Virtual points")

   
    ax[i,k].set_title(model.replace('_',' '))
    ax[i,k].set_xlabel('year')
    if renormalized:
        ax[i,k].set_ylim(0,6)
    else:
        ax[i,k].set_ylim(-3,3)


# ax[-2].legend(loc='center right',bbox_to_anchor=(3.2,0.5), fancybox=True, shadow=True)

handles, labels = ax[1,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3)
fig.tight_layout()
fig.suptitle('Results of experiment 3', y=1.01)
plt.savefig(path+'plot_exp3_temp.png', bbox_inches='tight', dpi=200)
plt.show()

# %%
