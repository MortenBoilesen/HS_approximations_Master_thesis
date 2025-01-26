#%% BASELINE MODEL
import numpy as np
from matplotlib import pyplot as plt
from utils import plot_with_samples, benchmark_ushaped, plot_gp_with_samples
import pickle

plt.rcParams.update({
    'font.size': 20.0,
    'axes.titlesize': 'medium'
})
textwidth = 5.9
plotwidth = 3.1*0.95*textwidth
plotheight = plotwidth*1.2


models = ['baseline_GP', 'HS_model',  'HS_2deriv_model', 'VP_model']

benchmark_function_names = ['flat', 'skew', 'parabola', 'abs', 'sine', 'step']
path = '6_Experiments/63_Ushaped_benchmark_functions/'
datapath = '6_Experiments/63_Ushaped_benchmark_functions/baseline_GP/results/'

x_pred = np.linspace( -5, 5, 100)

#%%
fig, ax = plt.subplots(len(benchmark_function_names), len(models), figsize=(plotwidth,plotheight))

for i in range(len(benchmark_function_names)):
    true_function = benchmark_ushaped(i, x_pred)
    y_train = np.load(datapath+f'{benchmark_function_names[i]}/ytrain.npy')
    y_test = np.load(datapath+f'{benchmark_function_names[i]}/ytest.npy')
    x_train = np.load(datapath+f'{benchmark_function_names[i]}/xtrain.npy')
    x_test = np.load(datapath+f'{benchmark_function_names[i]}/xtest.npy')

    for j, model in enumerate(models):
        if model != 'baseline_GP':        
            fpred = np.load(path+model+f'/results/{benchmark_function_names[i]}/10000_test_fpred.npy')
            sigma = np.load(path+model+f'/results/{benchmark_function_names[i]}/10000_test_sigma.npy')            
            plot_with_samples(ax[i,j], x_pred, fpred, sigma, num_samples = 10)
        
        else:
            fpred = np.load(path+model+f'/results/{benchmark_function_names[i]}/test_fpred.npy')
            mu = np.load(path+model+f'/results/{benchmark_function_names[i]}/test_mu.npy')
            std = np.load(path+model+f'/results/{benchmark_function_names[i]}/test_std.npy')
            plot_gp_with_samples(ax[i,j], x_pred, fpred, mu, std, num_samples = 10)

            
        ax[i,j].scatter(x_test,y_test,label="Testing data", color="tab:orange")
        ax[i,j].scatter(x_train,y_train,label="Training data", color="tab:blue")

        ax[i,j].plot(x_pred, true_function, color='black', label='True function', linestyle='--')
        if model == 'VP_model':
            with open(path+model+f'/results/{benchmark_function_names[i]}/model_selection_results.pkl', 'rb') as file:
                best_config = pickle.load(file)

            num_virtual_points = best_config['num_virtual']
            x_virtual = np.linspace(-5,5,num_virtual_points)
            lower, upper = ax[i,j].get_ylim()
            ax[i,j].scatter(x_virtual, -3*np.ones(x_virtual.shape)+0.07, marker='^', label="Virtual points")

        
        if j == 0:
            ax[i,j].set_ylabel(f'{benchmark_function_names[i]}\n' + 'f(x)')
        if i==5:
            ax[i,j].set_xlabel('x')
        if i == 0:
            ax[i,j].set_title(model.replace('_', ' '))

        # if j == 0:
        #     ax[i, j].text(-0.2, 0.5, benchmark_function_names[i].replace('sine','sin'), transform=ax[i, j].transAxes, va='center', rotation='vertical',fontsize = 16)
        # if i == 0:
        #     if model != 'HS_2deriv_model':
        #         ax[i,j].set_title(model.replace('_', ' '))
        #     else:
        #         ax[i,j].set_title('HS relaxed model')
        
        ax[i,j].set_ylim(-3,10)
        ax[i,j].set_xlabel('x')

# fig.suptitle('Results of experiment 2', y=0.99, fontsize=20)
# ax[0,-1].legend(loc='center right',bbox_to_anchor=(2.0,0.5), fancybox=True, shadow=True)

handles, labels = ax[0, -1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=22)  # Move the legend below the suptitle

fig.suptitle('Results of experiment 2', y=0.99)  # Adjust the y position of the suptitle
fig.tight_layout()
plt.savefig(path+'plot_exp2.png', bbox_inches='tight', dpi=500)
plt.show()



# %%
plt.rcParams.update({
    'font.size': 10.0,
    'axes.titlesize': 'medium'
})
textwidth = 5.9
plotwidth = textwidth
plotheight = plotwidth*0.6


fig, ax = plt.subplots(2, 3, figsize=(plotwidth, plotheight))

for i, func_name in enumerate(benchmark_function_names):
    true_function = benchmark_ushaped(i, x_pred)
    row, col = divmod(i, 3)
    ax[row, col].plot(x_pred, true_function, color='black', label='True function')
    ax[row, col].set_title(func_name.replace('sine', 'sin'))
    ax[row, col].set_ylim(np.min(true_function)-0.5, np.max(true_function)+0.5)
    # ax[row, col].legend()

plt.tight_layout()
plt.savefig(path + 'benchmark_functions_exp2.png', bbox_inches='tight')
plt.show()
# %%
### PLOT DERIVATIVE COMPONENTS

#%% plot of variance contribution from GP and concave part in experiment 2
plt.rcParams.update({
    'font.size': 20.0,
    'axes.titlesize': 'medium'
})
textwidth = 5.9
plotwidth = 4*0.95*textwidth
plotheight = 0.3*plotwidth

benchmark_function_names = ['flat', 'skew', 'parabola', 'abs', 'sine', 'step']
path = '6_Experiments/63_Ushaped_benchmark_functions/'
fig, axs = plt.subplots(6, 3, figsize=(plotwidth, plotheight * 6))

x_pred = np.linspace(-5, 5, 100)

for i, benchmark_function_name in enumerate(benchmark_function_names):
    resultpath = path+'HS_2deriv_model/results/' + benchmark_function_name +'/'

    gp_part = np.load(resultpath+'10000_test_gppart.npy')
    f_pred = np.load(resultpath+'10000_test_fpred.npy')
    sigma = np.load(resultpath+'10000_test_sigma.npy')
    x_train = np.load(resultpath +'10000_test_xtrain.npy')
    y_train = np.load(resultpath +'10000_test_ytrain.npy')
    convex_part = f_pred - gp_part
    results = np.load(resultpath+'10000_test_results.npy')
    true_function = benchmark_ushaped(i, x_pred)

    row = i

    axs[row, 0].scatter(x_train, y_train, label="Training data", color="tab:blue")
    plot_with_samples(ax=axs[row, 0], xpred=x_pred, fpred=f_pred, sigma=sigma, num_samples=100)
    axs[row, 0].plot(x_pred, true_function, linestyle=':', color='black', label='True function')
    axs[row, 0].set_title(f'{benchmark_function_name} - full HS relaxed model')
    axs[row, 0].set_xlabel('x')
    axs[row, 0].set_ylabel(f'{benchmark_function_name}')

    axs[row, 1].plot(x_pred, convex_part[:, 0], color='tab:pink', label='convex term', alpha=0.5)
    axs[row, 1].plot(x_pred, gp_part[:, 0], color='tab:cyan', label='GP term', alpha=0.5)
    for k in range(1, min(50, f_pred.shape[1])):
        axs[row, 1].plot(x_pred, convex_part[:, k], color='tab:pink', alpha=0.5)
        axs[row, 1].plot(x_pred, gp_part[:, k], color='tab:cyan', alpha=0.5)
    axs[row, 1].set_title('Samples from concave term and GP term')
    axs[row, 1].set_xlabel('x')
    axs[row, 1].set_ylabel('Value')

    axs[row, 2].plot(x_pred, np.var(convex_part, axis=1), label='Convex term', color='tab:pink')
    axs[row, 2].plot(x_pred, np.var(gp_part, axis=1), label='gp term', color='tab:cyan')
    axs[row, 2].set_title('Variance of components')
    axs[row, 2].set_xlabel('x')
    axs[row, 2].set_ylabel('Variance')

handles, labels = [], []
for ax in axs.flatten():
    for handle, label in zip(*ax.get_legend_handles_labels()):
        if label not in labels:
            handles.append(handle)
            labels.append(label)
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3)

fig.tight_layout()
plt.show()


# %%
