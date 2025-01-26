#%% BASELINE MODEL
import numpy as np
from matplotlib import pyplot as plt
from utils import plot_with_samples, benchmark_functions
import pickle

plt.rcParams.update({
    'font.size': 20.0,
    'axes.titlesize': 'medium'
})
models = ['baseline_GP', 'HS_model', 'VP_model', 'SDE_model' ]
path = '6_Experiments/61_Monotonic_benchmark_functions/'
datapath = '6_Experiments/61_Monotonic_benchmark_functions/baseline_GP/results/'

x_pred = np.linspace( 0, 10, 100)
#%%
benchmark_function_names = ['flat', 'sinusoidal', 'step', 'linear', 'exp', 'logical']
textwidth = 5.9
plotwidth = 3.1*0.95*textwidth
plotheight = plotwidth*1.2


fig, ax = plt.subplots(len(benchmark_function_names), len(models), figsize=(plotwidth,plotheight))
fig.subplots_adjust(top=0.88, bottom=0.1)  # Adjust the top and bottom spacing


for i in range(len(benchmark_function_names)):
    true_function = benchmark_functions(i, x_pred)
    y_train = np.load(datapath+f'{benchmark_function_names[i]}/10000_test_ytrain.npy')
    x_train = np.load(datapath+f'{benchmark_function_names[i]}/10000_test_xtrain.npy')

    # Generate data
    num_test = 100

    np.random.seed(1234*(i+1) + 18)
    x_test = np.random.uniform(0, 10, num_test)
    y_test = benchmark_functions(i, x_test) + np.random.normal(0, 1, num_test)

    for j, model in enumerate(models):
        try:
            fpred = np.load(path+model+f'/results/{benchmark_function_names[i]}/10000_test_fpred.npy')
            sigma = np.load(path+model+f'/results/{benchmark_function_names[i]}/10000_test_sigma.npy')
        except:
            fpred = np.load(path+model+f'/results/{benchmark_function_names[i]}/test_fpred.npy')
            sigma = np.load(path+model+f'/results/{benchmark_function_names[i]}/test_sigma.npy')


        plot_with_samples(ax[i,j], x_pred, fpred, sigma, alpha_samples=0.15, num_samples = 100)
        
        ax[i,j].plot(x_pred, true_function, color='black', label='True function', linestyle='--')
        if model == 'VP_model':
            with open(path+model+f'/results/{benchmark_function_names[i]}/model_selection_results.pkl', 'rb') as file:
                best_config = pickle.load(file)
            num_virtual_points = best_config['num_virtual']
            x_virtual = np.linspace(0,10,num_virtual_points)
            lower, upper = ax[i,j].get_ylim()
            ax[i,j].scatter(x_virtual, -3*np.ones(x_virtual.shape)+0.07, marker='^', label="Virtual points")

        ax[i,j].scatter(x_test,y_test,label="Test data", color="tab:orange")
        ax[i,j].scatter(x_train,y_train,label="Training data", color="tab:blue")


        if j == 0:
            ax[i,j].set_ylabel(f'{benchmark_function_names[i]}\n' + 'f(x)')
        if i==5:
            ax[i,j].set_xlabel('x')
        if i == 0:
            ax[i,j].set_title(model.replace('_', ' '))
        
        ax[i,j].set_ylim(-3,10)


handles, labels = ax[0, -2].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=22)

fig.suptitle('Results of experiment 1', y=0.99)  # Adjust the y position of the suptitle
fig.tight_layout()
plt.savefig(path+'plot_exp1.png', bbox_inches='tight', dpi =500)
plt.show()

#%%
textwidth = 5.9
plotwidth = 3.1*0.95*textwidth
plotheight = plotwidth*0.8

b_function_subset = [0, 3, 5]

fig, ax = plt.subplots(len(b_function_subset), len(models), figsize=(plotwidth,plotheight))
fig.subplots_adjust(top=0.88, bottom=0.1)  # Adjust the top and bottom spacing


for i, b_function_index in enumerate(b_function_subset):
    true_function = benchmark_functions(b_function_index, x_pred)
    y_train = np.load(datapath+f'{benchmark_function_names[b_function_index]}/10000_test_ytrain.npy')
    x_train = np.load(datapath+f'{benchmark_function_names[b_function_index]}/10000_test_xtrain.npy')

    # Generate data
    num_test = 100

    np.random.seed(1234*(i+1) + 18)
    x_test = np.random.uniform(0, 10, num_test)
    y_test = benchmark_functions(b_function_index, x_test) + np.random.normal(0, 1, num_test)

    for j, model in enumerate(models):
        try:
            fpred = np.load(path+model+f'/results/{benchmark_function_names[b_function_index]}/10000_test_fpred.npy')
            sigma = np.load(path+model+f'/results/{benchmark_function_names[b_function_index]}/10000_test_sigma.npy')
        except:
            fpred = np.load(path+model+f'/results/{benchmark_function_names[b_function_index]}/test_fpred.npy')
            sigma = np.load(path+model+f'/results/{benchmark_function_names[b_function_index]}/test_sigma.npy')


        plot_with_samples(ax[i,j], x_pred, fpred, sigma, alpha_samples=0.15, num_samples = 100)
        
        ax[i,j].plot(x_pred, true_function, color='black', label='True function', linestyle='--')
        if model == 'VP_model':
            with open(path+model+f'/results/{benchmark_function_names[b_function_index]}/model_selection_results.pkl', 'rb') as file:
                best_config = pickle.load(file)
            num_virtual_points = best_config['num_virtual']
            x_virtual = np.linspace(0,10,num_virtual_points)
            lower, upper = ax[i,j].get_ylim()
            ax[i,j].scatter(x_virtual, -3*np.ones(x_virtual.shape)+0.07, marker='^', label="Virtual points")

        ax[i,j].scatter(x_test,y_test,label="Test data", color="tab:orange")
        ax[i,j].scatter(x_train,y_train,label="Training data", color="tab:blue")


        if j == 0:
            ax[i,j].set_ylabel(f'{benchmark_function_names[b_function_index]}\n' + 'f(x)')
        if i==5:
            ax[i,j].set_xlabel('x')
        if i == 0:
            ax[i,j].set_title(model.replace('_', ' '))
        
        ax[i,j].set_ylim(-3,10)


handles, labels = ax[0, -2].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.07), ncol=4, fontsize=22)

fig.suptitle('Selected results of experiment 1', y=0.99)  # Adjust the y position of the suptitle
fig.tight_layout()
plt.savefig(path+'plot_exp1_sub.png', bbox_inches='tight', dpi =500)
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
    true_function = benchmark_functions(i, x_pred)
    row, col = divmod(i, 3)
    ax[row, col].plot(x_pred, true_function, color='black', label='True function')
    ax[row, col].set_title(func_name)
    ax[row, col].set_ylim(np.min(true_function)-0.5, np.max(true_function)+0.5)
    # ax[row, col].legend()

plt.tight_layout()
plt.savefig(path + 'benchmark_functions_exp1.png', bbox_inches='tight', dpi=200)
plt.show()