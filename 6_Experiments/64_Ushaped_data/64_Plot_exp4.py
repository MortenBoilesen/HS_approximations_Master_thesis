#%% BASELINE MODEL
import numpy as np
from matplotlib import pyplot as plt
from utils import plot_with_samples, benchmark_ushaped, plot_gp_with_samples
import pickle

r = 4

plt.rcParams.update({
    'font.size': 18.0,
    'axes.titlesize': 'medium'
})
textwidth = 5.9

models = ['baseline_GP', 'HS_model', 'VP_model']
path = f'6_Experiments/64_Ushaped_data/Runs/Run_{r}/'
# fractions = [0.5, 0.7272727272727273, 0.8888888888888888, 1.0, 2.0, 4.0]
# fractions = [0.5, 0.8888888888888888, 1.0, 2.0, 4.0]
fractions = [0.8888888888888888, 2.0]

plotwidth = 2.5*0.95*textwidth
plotheight = plotwidth*0.6


# Load data
datapath = path + '/00data/'
x_train_tot = np.load(datapath + 'x_train_shuffled.npy')
y_train_tot = np.load(datapath + 'y_train_shuffled.npy')
num_total = len(x_train_tot)

x_test = np.load(datapath + 'x_test.npy')
y_test = np.load(datapath + 'y_test.npy')
x_pred = np.linspace( -1.8, 1.8, 100)

upper = 3.5
lower = -1.1
pad = 0.07

renormalized = True

if renormalized:
    x_mean = 43.0
    x_std = 15.874507866387544

    x_train_tot = x_train_tot*x_std + x_mean
    x_test = x_test*x_std + x_mean
    x_pred = x_pred*x_std + x_mean
    
    y_train_tot /= 100
    y_test /= 100 

    upper /= 100
    lower /= 100
    pad /= 100


fig, ax = plt.subplots(len(fractions), len(models), figsize=(plotwidth,plotheight))

for i, fraction in enumerate(fractions):
    num_train = int(fraction* num_total)//8
    x_train = x_train_tot[:num_train]
    y_train = y_train_tot[:num_train]

    for j, model in enumerate(models):
        resultpath = path + model +f'/{fraction}/'
        fpred = np.load(resultpath + 'test_fpred.npy')
        sigma = np.load(resultpath + 'test_sigma.npy')
        if renormalized:
            fpred /= 100
            sigma /= 100
        
        plot_with_samples(ax[i,j], x_pred, fpred, sigma, num_samples = 100, alpha_samples=0.3)
        
            
        if model == 'VP_model':
            with open(resultpath + 'model_selection_results.pkl', 'rb') as file:
                best_config = pickle.load(file)
            
            num_virtual_points = best_config['num_virtual'] 
            title = f'num virtual = {num_virtual_points}'
            x_virtual = np.linspace(-1.8,1.8,num_virtual_points)
            if renormalized:
                x_virtual = x_virtual*x_std + x_mean
                ax[i,j].scatter(x_virtual, lower*np.ones(x_virtual.shape) + pad, marker='^', label="Virtual points")
        
        elif model == 'HS_model':
            with open(resultpath + 'model_selection_results.pkl', 'rb') as file:
                best_config = pickle.load(file)
            m = best_config["m"]
            L = best_config["L"]
            title = f'm = {m}, L = {L}'
        else:
            title = ''



        ax[i,j].scatter(x_test,y_test,label="Test data", color="tab:orange", zorder=10)
        ax[i,j].scatter(x_train,y_train,label="Training data", color="tab:blue", zorder=10)

        
        
        if j == 0:
            ax[i,j].set_ylabel(f'{num_train} training points\n' + 'Risk of depression')
        if i==1:
            ax[i,j].set_xlabel('Age')

        ax[i,j].set_title((model.replace('_', ' ') +'\n')*(i==0) + title)
        ax[i,j].set_ylim(lower,upper)


# fig.suptitle('Results of experiment 4', y=1.0, fontsize=20)
# ax[0,-1].legend(loc='center right',bbox_to_anchor=(2.0,0.5), fancybox=True, shadow=True)

handles, labels = ax[0, -1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=22)
fig.suptitle(f'Results of experiment 4. Run {r}', y=0.96)  # Adjust the y position of the suptitle

fig.tight_layout()
plt.savefig('6_Experiments/64_Ushaped_data/plot_exp4.png',bbox_inches='tight')
plt.show()


