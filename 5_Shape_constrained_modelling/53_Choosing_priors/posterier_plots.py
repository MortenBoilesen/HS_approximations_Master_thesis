#%%
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import plot_with_samples
plt.rcParams.update({'font.size': 12})

path = 'experiment_2_F_prior/'
path  = ''
num_chains = 3
num_samples = 1000
xtrain = np.load(path+'data/xtrain.npy')
ytrain = np.load(path+'data/ytrain.npy')
xtest = np.load(path+'data/xtest.npy')
ytest = np.load(path+'data/ytest.npy')

#%%
### GAUSSIAN PRIORS
f0_prior = F0_prior = 'standardnormal'
resultpath = path + f'results/'+f0_prior+'_'+F0_prior + '/'

print(resultpath)

num_pred = 100
xpred = np.linspace(-2.5,2.5,num_pred)

fpred = np.load(resultpath+'fpred.npy')
ftest = np.load(resultpath+'ftest.npy')
ftrain = np.load(resultpath+'ftrain.npy')
sigma = np.load(resultpath+'sigma.npy')
lpd = np.load(resultpath+'lpd.npy')

fig, ax  = plt.subplots(1, figsize=(6,4))

plot_with_samples(ax=ax, xpred=xpred, fpred=fpred,sigma=sigma, num_samples=100, color_samples="tab:green")
ax.scatter(xtrain, ytrain,label='train')
ax.scatter(xtest, ytest,label='test')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'ELPD = {lpd:.5}',fontsize=14)
plt.legend(loc='upper center')
plt.savefig(path + 'figures/standard_gaussian.png')
plt.show()


#%%
### DIFFERENT PRIORS

f0_prior_list = ['normal','gamma','lognormal']
F0_prior_list = ['normal','studentt']

f0_prior_list_labels = ['negative halfnormal','negative gamma','negative lognormal']


fig, ax = plt.subplots(2,3, figsize=(15, 9))

for i,F0_prior in enumerate(F0_prior_list):
    for j,f0_prior in enumerate(f0_prior_list):
        resultpath = path + f'results/'+f0_prior+'_'+F0_prior + '/'
        print(resultpath)

        ftest = np.load(resultpath+'ftest.npy')
        ftrain = np.load(resultpath+'ftrain.npy')
        fpred = np.load(resultpath+'fpred.npy')
        sigma = np.load(resultpath+'sigma.npy')
        lpd = np.load(resultpath+'lpd.npy')

        plot_with_samples(ax=ax[i,j], xpred=xpred, fpred=fpred, sigma=sigma, color_samples='tab:green', num_samples=100)
        ax[i,j].scatter(xtrain, ytrain,label='train')
        ax[i,j].scatter(xtest, ytest,label='test')
        ax[i,j].set_xlabel('x')
        ax[i,j].set_ylabel('y')
        ax[i,j].set_title(f'ELPD = {lpd:.5}',fontsize=14)

        # Add text above each column
        if i == 0:
            ax[i,j].text(0.5, 1.1, 'f0 prior: '+f0_prior_list_labels[j], transform=ax[i,j].transAxes, ha='center', fontsize=16)
            if j ==1:
                ax[i,j].legend(loc='upper center')

        if j == 0: 
            ax[i, j].text(-0.2, 0.5, 'F0 prior: '+F0_prior, transform=ax[i, j].transAxes, va='center', rotation='vertical',fontsize = 16)
            fig.subplots_adjust(hspace=0.4)  # Add more padding under the rows

fig.tight_layout()
plt.savefig(path + 'figures/prior_experiments_temp.png')
plt.show()


#%%
### wITH MEAN
num_chains = 3
f0_prior = 'lognormal'
F0_prior = 'studentt'
resultpath = path + f'results/'+f0_prior+'_'+F0_prior + '_with_mean/'

print(resultpath)

num_pred = 100
xpred = np.linspace(-2.5,2.5,num_pred)

fpred = np.load(resultpath+'fpred.npy')
ftest = np.load(resultpath+'ftest.npy')
ftrain = np.load(resultpath+'ftrain.npy')
sigma = np.load(resultpath+'sigma.npy')
lpd = np.load(resultpath+'lpd.npy')

fig, ax  = plt.subplots(1, figsize=(6,4))

plot_with_samples(ax=ax, xpred=xpred, fpred=fpred,sigma=sigma, num_samples=100, color_samples="tab:green")
ax.scatter(xtrain, ytrain,label='train', zorder=12)
ax.scatter(xtest, ytest,label='test', zorder=12)

num_train = len(xtrain)
sort_idx = np.argsort(xtrain)
index = sort_idx[:(num_train // 4)]
# Create the subset
x_train_subset = xtrain[index]
y_train_subset = ytrain[index]
X_b = np.c_[np.ones((x_train_subset.shape[0], 1)), x_train_subset]

# Calculate the optimal theta using the normal equation
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train_subset)
ax.plot(xpred, theta_best[0] + xpred*theta_best[1], linestyle='-.', color='black', zorder=10, label='Regression heuristic')
ax.set_ylim(bottom=-5.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'ELPD = {lpd:.5}',fontsize=14)
plt.legend(loc='upper center')
plt.savefig(path + 'figures/with_intercept.png', bbox_inches='tight')
plt.show()
