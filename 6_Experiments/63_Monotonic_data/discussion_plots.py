#%%
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from utils import plot_with_samples
import pandas as pd
plt.rcParams.update({
    'font.size': 10.0,
    'axes.titlesize': 'medium'
})

textwidth = 5.9
plotwidth = 5.9*0.95
plotheight = plotwidth*0.6

######### VP PLOT
path = '5_Shape_constrained_modelling/53_Relaxing_convexity/'
path = ''
datapath = path + '00data/Fertility_rate/'
resultpath = path + 'VP_model/results/discussion/'

result_df = pd.DataFrame(index=['RMSE', 'ELPD'])

x_train = np.load(datapath + 'x_train.npy')
y_train = np.load(datapath + 'y_train.npy')
x_test = np.load(datapath + 'x_test.npy')
y_test = np.load(datapath + 'y_test.npy')

x_pred = np.linspace(x_train.min(),x_test.max(),100)
pred_start = len(x_pred)//2
x_pred = x_pred[pred_start:]
colors = ['blue', 'green', 'orange', 'purple']

fig, ax = plt.subplots(1,1, figsize=(plotwidth, plotheight))
for k in range(4):
    f_pred = np.load(resultpath + f'{k+1}_fpred.npy')
    sigma = np.load(resultpath + f'{k+1}_sigma.npy') 
    mu = np.mean(f_pred,axis=-1)

    plot_with_samples(ax, x_pred,f_pred[pred_start:], sigma, color=colors[k], alpha=0.1, color_samples=colors[k], color_CI='tab:'+colors[k], title=f'{k+1} virtual points.')

    results = np.load(resultpath + f'{k+1}_virtual_results.npy')
    result_df[f'{k+1} virtual point' + 's'*(k+1 > 1)] = results

ax.scatter(x_train[x_train >= x_pred.min()],y_train[x_train >= x_pred.min()],label='training data')
ax.scatter(x_test,y_test,label='test data')

handles, labels = ax.get_legend_handles_labels()
new_handles = [handles.pop(-2)]
new_labels = [labels.pop(-2)]
new_handles.append(handles.pop(-1))
new_labels.append(labels.pop(-1))

k = 1
for i,label in enumerate(labels):
    if label.startswith('Posterior'):
        string = f'{k} virtual point' + 's'*(k > 1)
        new_labels.append(string +', RMSE = ' + str(np.round(result_df[string]['RMSE'],2)))
        new_handles.append(handles[i])
        k+=1

ax.legend(loc='lower left', handles=new_handles, labels=new_labels, ncol=1)
ax.set_title('VP model trained with 4,3,2 and 1 virtual points')
year_std = 18.184242262647807
year_mean = 1991

y_mean = 3.4063193866029415
y_std = 0.9264302087414242

xticks = ax.get_xticks()[1:(len(ax.get_xticks())-1)]
ax.set_xticks(ticks=xticks, labels= [int(label) for label in xticks*year_std + year_mean])
yticks = ax.get_yticks()[1:(len(ax.get_yticks())-1)]
ax.set_yticks(ticks=yticks, labels= [round(label,2) for label in yticks*y_std + y_mean])
ax.set_xlabel('year')
ax.set_ylabel('fertiliy rate')

plt.tight_layout()
plt.savefig(path + 'VP_discussion_plot.png', bbox_inches='tight', dpi=200)

# ## Generate table
latex = '\\begin{table}[] \n \\centering\n'
styled_df = result_df.style
styled_df = styled_df.format("{:.3f}")
latex += styled_df.to_latex(hrules=True)
latex = latex.replace('\\end{tabular}\n', '')
latex = latex.replace('toprule', 'hline').replace('midrule', 'hline').replace('bottomrule', 'hline')
latex = latex.replace('_', '\_')
latex += '\\end{tabular}\n\\caption{}\n\\label{tab:exp3_vp_discussion}\n\\end{table}\n'

print(latex)

############# HS PLOT
#     parameter_names = ['f0', 'f0_param', 'kappa', 'scale', 'sigma', 'alpha']
# plotheight = plotwidth*0.6
params = np.load(path + 'HS_model/results/params_and_rhat/10000_test_params.npy')

xtrain = np.load(path + '00data/Fertility_rate/x_train.npy')
ytrain = np.load(path + '00data/Fertility_rate/y_train.npy')
xtest = np.load(path + '00data/Fertility_rate/x_test.npy')
ytest = np.load(path + '00data/Fertility_rate/y_test.npy')

xpred_true = np.linspace(np.min(xtrain), np.max(xtest), 100)
f_true = np.load(path + 'HS_model/results/10000_test_fpred.npy')

f0 = params[0,:]
kappa = params[2,:]
scale = params[3,:]
sigma = params[4,:]
alpha = params[5:,:]

num_pred = 100
L = 3
m = 10
xpred = np.linspace(-4,4,num_pred)

def spectral_density(omega, theta_k):
    kappa = theta_k[0]
    scale = theta_k[1]
    return kappa**2*np.sqrt(2*np.pi*scale**2)*np.exp(omega*omega *(-0.5)*scale**2)

def lambda_sqrt(j, L):
    return j*np.pi/(2*L)

def phi(x, j, L):
    return 1/np.sqrt(L)*np.sin(lambda_sqrt(j, L)*(x + L))*(np.abs(x) <= L)

def Psi(x, i, j, L):
    lam_diff = lambda_sqrt(i, L) - lambda_sqrt(j, L)
    lam_sum = lambda_sqrt(i, L) + lambda_sqrt(j, L)


    xpL = x + L
    xpL[np.where(x > L)] = 2*L
    xpL[np.where(x < -L)] = 0

    if i == j:
        return 0.5*(xpL) - 1./(4*lambda_sqrt(j, L))*np.sin(2*lambda_sqrt(j, L)*(xpL))

    else:
        return 1./(2*lam_diff)*np.sin(lam_diff*(xpL)) - 1./(2*lam_sum)*np.sin(lam_sum*(xpL));

PSI_array = np.zeros((m, m, num_pred))
phi_array = np.zeros((m, num_pred))
Lambda = np.zeros((m, 40000))

theta_k = np.stack([kappa,scale], axis = 0)

for i in range(m):
    phi_array[i,:] = phi(xpred, i+1, L)
    Lambda[i,:] = np.sqrt(spectral_density( lambda_sqrt(i+1, L), theta_k ))
    for j in range(m):
        PSI_array[i,j] = Psi(xpred, i+1, j+1, L)/L

alpha *= Lambda

fs = np.zeros((num_pred, 40000))
gp = np.zeros((num_pred, 40000))

for i in range(m):
    gp += alpha[i]*phi_array[i][:,None]
    for j in range(m):
        fs += alpha[i]*alpha[j]*PSI_array[i,j][:,None]

fs += f0
fs = -fs


textwidth = 5.9
fig, ax = plt.subplots(1,1,figsize=(plotwidth, plotheight))
plot_with_samples(ax,xpred,fs,sigma,num_samples=100)
# ax.plot(xpred, fs[:,:100], color='tab:red', alpha=0.2)
# ax.plot(xpred_true, f_true[:,:100], color='tab:blue')    

ax.scatter(xtrain,ytrain,label='train')
ax.scatter(xtest,ytest, label='test')
ax.legend(ncol=1)
ax.set_title('HS model on all of [-L,L]')
xticks = ax.get_xticks()[1:(len(ax.get_xticks())-1)]
ax.set_xticks(ticks=xticks, labels= [int(label) for label in xticks*year_std + year_mean])
yticks = ax.get_yticks()[1:(len(ax.get_yticks())-1)]
ax.set_yticks(ticks=yticks, labels= [round(label,2) for label in yticks*y_std + y_mean])
ax.set_xlabel('year')
ax.set_ylabel('fertiliy rate')

plt.tight_layout()
plt.savefig(path + 'HS_discussion_plt.png', dpi=200, bbox_inches='tight')

