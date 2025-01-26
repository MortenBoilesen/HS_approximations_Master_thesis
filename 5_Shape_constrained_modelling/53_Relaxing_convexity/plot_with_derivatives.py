#%%
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import plot_with_samples

path = ''
# path = '6_Experiments/63_Monotonic_data/'
num_chains = 3
num_samples = 1000

N_samples = 20

models = ['concave', 'add_GP', 'add_GP_deriv', 'add_GP_second_deriv']
num_pred = 100 
# x_pred = np.linspace(-2.5,2.5, num_pred)
x_pred = np.linspace(-5,5, num_pred)

datapath = path + '00data/'
x_train = np.load(datapath+'xtrain.npy')
y_train = np.load(datapath+'ytrain.npy')
x_test = np.load(datapath+'xtest.npy')
y_test = np.load(datapath+'ytest.npy')


import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(4, 4, figure=fig)

# plot baseline
resultpath = path+'results/'
sigma = np.load(resultpath+'baseline_GP/sigma.npy')
f_pred = np.load(resultpath+'baseline_GP/fpred.npy')
results  = np.load(resultpath + 'baseline_GP/results.npy')
RMSE = results[0] 
lpd = results[1]

ax0 = fig.add_subplot(gs[1, 0])
plot_with_samples(ax=ax0, xpred=x_pred, fpred=f_pred, sigma=sigma,num_samples=10)
ax0.scatter(x_train,y_train,label="Training data", color="tab:blue")
ax0.scatter(x_test,y_test,label="Test data", color="tab:orange")

ax0.set_xlabel('x')
ax0.set_ylabel('y')
ax0.set_title(f'Baseline GP, lpd = {lpd:.4}, \nRMSE = {RMSE:.4}')


# loop though the rest
for k, model in enumerate(models):

    f_pred = np.load(resultpath+model+'/fpred.npy')
    results = np.load(resultpath+model+'/results.npy')
    RMSE = results[0] 
    lpd = results[1]
    ax0 = fig.add_subplot(gs[0, k])
    plot_with_samples(ax=ax0,xpred=x_pred, fpred=f_pred, sigma=sigma, num_samples=10)

    if model == 'add_GP':
        gppred = np.load(resultpath+model+f'/gppred.npy')
        u_pred = f_pred + gppred
        ax1 = fig.add_subplot(gs[1, k])
        ax1.plot(x_pred,-gppred[:,0], linewidth=0.5, alpha=0.5, color='tab:cyan', label='GP part')
        ax1.plot(x_pred,-gppred[:,1:N_samples], linewidth=0.5, alpha=0.5, color='tab:cyan')

        ax1.plot(x_pred,u_pred[:,0], linewidth=0.5, alpha=0.5, color='tab:pink', label='concave part')
        ax1.plot(x_pred,u_pred[:,1:N_samples], linewidth=0.5, alpha=0.5, color='tab:pink')
        ax1.set_title('Components')

    if model == 'add_GP_deriv':
        Gppred = np.load(resultpath+model+f'/gppred_int.npy')
        u_pred = f_pred + Gppred
        ax1 = fig.add_subplot(gs[1, k])
        ax1.plot(x_pred,-Gppred[:,0], linewidth=0.5, alpha=0.5, color='tab:cyan', label='GP part')
        ax1.plot(x_pred,-Gppred[:,1:N_samples], linewidth=0.5, alpha=0.5, color='tab:cyan')

        ax1.plot(x_pred,u_pred[:,0], linewidth=0.5, alpha=0.5, color='tab:pink', label='concave part')
        ax1.plot(x_pred,u_pred[:,1:N_samples], linewidth=0.5, alpha=0.5, color='tab:pink')
        ax1.set_title('Components')

        fpred_d = np.load(resultpath+model+'/fpred_d.npy')
        gppred = np.load(resultpath+model+'/gppred.npy')
        m_pred = fpred_d + gppred

        ax2 = fig.add_subplot(gs[2, k])
        ax2.plot(x_pred,-gppred[:,0], linewidth=0.5, alpha=0.5, color='tab:cyan', label='GP part')
        ax2.plot(x_pred,-gppred[:,1:N_samples], linewidth=0.5, alpha=0.5, color='tab:cyan')

        ax2.plot(x_pred,m_pred[:,0], linewidth=0.5, alpha=0.5, color='tab:pink', label='monotone part')
        ax2.plot(x_pred,m_pred[:,1:N_samples], linewidth=0.5, alpha=0.5, color='tab:pink')
        ax2.plot(x_pred, fpred_d.mean(axis=1))
        ax2.set_title('Components of first derivative')

    if model == 'add_GP_second_deriv':
        GPpred = np.load(resultpath+model+f'/gppred_intint.npy')
        u_pred = f_pred + GPpred
        ax1 = fig.add_subplot(gs[1, k])
        ax1.plot(x_pred,-GPpred[:,0], linewidth=0.5, alpha=0.5, color='tab:cyan', label='GP part')
        ax1.plot(x_pred,-GPpred[:,1:N_samples], linewidth=0.5, alpha=0.5, color='tab:cyan')

        ax1.plot(x_pred,u_pred[:,0], linewidth=0.5, alpha=0.5, color='tab:pink', label='concave part')
        ax1.plot(x_pred,u_pred[:,1:N_samples], linewidth=0.5, alpha=0.5, color='tab:pink')
        ax1.plot(x_pred, np.mean(u_pred - GPpred, axis=1))
        ax1.set_ylim
        ax1.set_title('Components')
        
        fpred_d = np.load(resultpath+model+'/fpred_d.npy')
        Gppred = np.load(resultpath+model+'/gppred_int.npy')
        m_pred = fpred_d + Gppred

        ax2 = fig.add_subplot(gs[2, k])
        ax2.plot(x_pred,-Gppred[:,0], linewidth=0.5, alpha=0.5, color='tab:cyan', label='GP part')
        ax2.plot(x_pred,-Gppred[:,1:N_samples], linewidth=0.5, alpha=0.5, color='tab:cyan')

        ax2.plot(x_pred,m_pred[:,0], linewidth=0.5, alpha=0.5, color='tab:pink', label='monotonic part')
        ax2.plot(x_pred,m_pred[:,1:N_samples], linewidth=0.5, alpha=0.5, color='tab:pink')

        ax2.plot(x_pred, fpred_d.mean(axis=1))
        ax2.set_title('Components of first derivative')
        
        fpred_d2 = np.load(resultpath+model+'/fpred_d2.npy')
        gppred = np.load(resultpath+model+'/gppred.npy')
        n_pred = fpred_d2 + gppred

        ax3 = fig.add_subplot(gs[3, k])
        ax3.plot(x_pred,-gppred[:,0], linewidth=0.5, alpha=0.5, color='tab:cyan', label='GP part')
        ax3.plot(x_pred,-gppred[:,1:N_samples], linewidth=0.5, alpha=0.5, color='tab:cyan')

        ax3.plot(x_pred,n_pred[:,0], linewidth=0.5, alpha=0.5, color='tab:pink', label='negative part')
        ax3.plot(x_pred,n_pred[:,1:N_samples], linewidth=0.5, alpha=0.5, color='tab:pink')
        ax3.plot(x_pred, fpred_d2.mean(axis=1))
        ax3.set_title('Components of second derivative')
        

    ax0.scatter(x_train,y_train,label="Training data", color="tab:blue")
    ax0.scatter(x_test,y_test,label="Test data", color="tab:orange")

    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    ax0.set_title(model+f', lpd = {lpd:.4},\nRMSE = {RMSE:.4}')

custom_handles = [
    Line2D([], [], color='tab:cyan', label='gp part', linewidth=2),
    Line2D([], [], color='tab:pink', label='model part', linewidth=2),
]

# Create a common legend for all the plots
handles, labels = ax0.get_legend_handles_labels()
handles.extend(custom_handles)
labels.extend([handle.get_label() for handle in custom_handles])
fig.legend(handles, labels, loc='lower left', ncol=3)



fig.tight_layout()
plt.savefig(path + 'model_plot.png')
plt.show()

# %%

plt.rcParams.update({
    'font.size': 20.0,
    'axes.titlesize': 'medium'
})
textwidth = 5.9
plotwidth = 3*0.95*textwidth
plotheight = plotwidth*0.4


fig, axs = plt.subplots(1, 3, figsize=(plotwidth, plotheight))

# First subplot
axs[0].plot(x_pred, np.var(u_pred, axis=1), label='model part', color='tab:pink')
axs[0].plot(x_pred, np.var(GPpred, axis=1), label='gp part', color='tab:cyan')
axs[0].set_title('Variance of components')
axs[0].set_xlabel('x')
axs[0].set_ylabel('Variance')
axs[0].legend()

# Second subplot
axs[1].plot(x_pred, np.var(m_pred, axis=1), label='model part', color='tab:pink')
axs[1].plot(x_pred, np.var(Gppred, axis=1), label='gp part', color='tab:cyan')
axs[1].set_title('Variance of the \n first derivative components')
axs[1].set_xlabel('x')
axs[1].set_ylabel('Variance')
axs[1].legend()

# Third subplot
axs[2].plot(x_pred, np.var(n_pred, axis=1), label='model part', color='tab:pink')
axs[2].plot(x_pred, np.var(gppred, axis=1), label='gp part', color='tab:cyan')
axs[2].set_title('Variance of the \n second derivative components')
axs[2].set_xlabel('x')
axs[2].set_ylabel('Variance')
axs[2].legend()

fig.tight_layout()
plt.savefig(path + 'relaxed_model_plot_variance.png')
plt.show()

#%%
##### PLOT VARIANCE OF COMPONENTS


#%% plot of the variance contributions from the concave and GP part for model 3
plt.rcParams.update({
    'font.size': 20.0,
    'axes.titlesize': 'medium'
})
textwidth = 5.9
plotwidth = 3*0.95*textwidth
plotheight = plotwidth*0.4
path = '6_Experiments/63_Monotonic_data/'
resultpath = path+'results/'
model = 'add_GP_second_deriv'

x_pred = np.linspace(-5, 5, 100)
f_pred = np.load(resultpath+model+f'/fpred.npy')
GPpred = np.load(resultpath+model+f'/gppred_intint.npy')
u_pred = f_pred + GPpred


fpred_d = np.load(resultpath+model+'/fpred_d.npy')
Gppred = np.load(resultpath+model+'/gppred_int.npy')
m_pred = fpred_d + Gppred

fpred_d2 = np.load(resultpath+model+'/fpred_d2.npy')
gppred = np.load(resultpath+model+'/gppred.npy')
n_pred = fpred_d2 + gppred

fig, axs = plt.subplots(1, 3, figsize=(plotwidth, plotheight))

# First subplot
axs[0].plot(x_pred, np.var(u_pred, axis=1), label='model part', color='tab:pink')
axs[0].plot(x_pred, np.var(GPpred, axis=1), label='gp part', color='tab:cyan')
axs[0].set_title('Variance of components')
axs[0].set_xlabel('x')
axs[0].set_ylabel('Variance')
axs[0].legend()

# Second subplot
axs[1].plot(x_pred, np.var(m_pred, axis=1), label='model part', color='tab:pink')
axs[1].plot(x_pred, np.var(Gppred, axis=1), label='gp part', color='tab:cyan')
axs[1].set_title('Variance of the \n first derivative components')
axs[1].set_xlabel('x')
axs[1].set_ylabel('Variance')
axs[1].legend()

# Third subplot
axs[2].plot(x_pred, np.var(n_pred, axis=1), label='model part', color='tab:pink')
axs[2].plot(x_pred, np.var(gppred, axis=1), label='gp part', color='tab:cyan')
axs[2].set_title('Variance of the \n second derivative components')
axs[2].set_xlabel('x')
axs[2].set_ylabel('Variance')
axs[2].legend()

fig.tight_layout()
plt.savefig(path + 'relaxed_model_plot_variance.png')
plt.show()
