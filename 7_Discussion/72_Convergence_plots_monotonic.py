#%%
import numpy as np
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import pickle
import matplotlib.gridspec as gridspec
from utils import benchmark_functions
from matplotlib.lines import Line2D
import os
path = '7_Discussion/' 
os.makedirs(path + 'exp1', exist_ok=True)

plt.rcParams.update({
    'font.size': 16.0,
    'axes.titlesize': 'medium'
})

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

    if i == j:
        return 0.5*(xpL) - 1./(4*lambda_sqrt(j, L))*np.sin(2*lambda_sqrt(j, L)*(xpL))

    else:
        return 1./(2*lam_diff)*np.sin(lam_diff*(xpL)) - 1./(2*lam_sum)*np.sin(lam_sum*(xpL));

def pred(m, L, theta_k, alpha, f0, x_pred):
    kappa, scale = theta_k
    num_pred = len(x_pred)
    num_samples = len(kappa)

    PSI_array = np.zeros((m, m, num_pred))
    phi_array = np.zeros((m, num_pred))
    Lambda = np.zeros((m, num_samples))

    theta_k = np.stack([kappa,scale], axis = 0)

    for i in range(m):
        phi_array[i,:] = phi(xpred, i+1, L)
        Lambda[i,:] = np.sqrt(spectral_density( lambda_sqrt(i+1, L), theta_k ))
        for j in range(m):
            PSI_array[i,j] = Psi(xpred, i+1, j+1, L)/L

    alpha *= Lambda
    fs = np.zeros((num_pred, num_samples))
    for i in range(m):
        for j in range(m):
            fs += alpha[i]*alpha[j]*PSI_array[i,j][:,None]

    fs += f0

    return fs

exp = '6_Experiments/61_Monotonic_benchmark_functions'
function = 'flat'
b_function_index = 0
#11 good, 2 bad

path = exp +f'/HS_model/results/{function}/'


num_chains = 4
num_samples  = 10000

params = np.load(path + 'params_and_rhat/10000_test_params.npy')
rhats = np.load(path + 'params_and_rhat/10000_test_rhat.npy')

xpred = np.linspace(0,10,100)


textwidth = 5.9
plotwidth = 1.5*0.95*textwidth
plotheight = plotwidth

parameter_names = ['f0', 'f0_param', 'kappa', 'scale', 'sigma', 'alpha_0', 'alpha_1']

r = 10

fig, ax = plt.subplots(4,2,figsize=(plotwidth, plotheight))
# gs = gridspec.GridSpec(2, 3, figure=fig) 

for p in range(7):
    i = p // 2
    j = p % 2
    param = params[r,p]

    for c in range(3,-1,-1):
        chain_idx = np.arange(c,num_chains*num_samples, num_chains)
        ax[i,j].plot(param[chain_idx], label = f'Chain {c+1}, Rhat = {np.round(np.max(rhats[r,c]),2)}', alpha=0.5)


    ax[i,j].set_title(parameter_names[p])

handles, labels = ax[i,j].get_legend_handles_labels()
handles.reverse()
labels.reverse()
j += 1
ax[i,j].legend(labels=labels, handles=handles, loc='center')
ax[i,j].axis('off')

# plt.suptitle(f', Run {r}, chain {c+1}, '  +  '$\mathrm{max}\left(\hat{R}\\right) = $ ' + f'${max_rhat:.2f}$')
plt.tight_layout()
plt.show()


#%%

def color_map_color(value, cmap_name='cool', vmin=0, vmax=1):
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgb = cmap(norm(abs(value)))[:3]  # will return rgba, we take only first 3 so we get rgb
    color = matplotlib.colors.rgb2hex(rgb)
    return color

s = np.arange(num_samples)

# Map the normalized values to colors
colors = [color_map_color(value, vmin=0, vmax=num_samples) for value in s]

textwidth = 5.9
plotwidth = 2.5*0.95*textwidth
plotheight = plotwidth*0.65

def convergece_plots(params, m, L, r, c, marked_centers = None, tol=0.15):
    fig = plt.figure(figsize=(plotwidth, plotheight))
    gs = gridspec.GridSpec(2, 3, figure=fig) 

    chain_idx = np.arange(c,num_chains*num_samples, num_chains)

    f0 = params[r,0,chain_idx]
    f0_param = params[r,1,chain_idx]
    kappa = params[r,2,chain_idx]
    scale = params[r,3,chain_idx]
    sigma = params[r,4,chain_idx]
    alpha = params[r,5:,chain_idx].T

    seed = 1234*(b_function_index+1) + r
    np.random.seed(seed)
    x_train = np.random.uniform(0, 10, 15)
    y_train = benchmark_functions(b_function_index, x_train) + np.random.normal(0, 1, 15)

    fpred = pred(m=m, L=L, theta_k=[kappa,scale],alpha=alpha, f0=f0, x_pred=xpred)

    # SUB 1
    gs00 =  gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0,0], height_ratios=[0.2,0.7], hspace=0.5)
    ax1 = fig.add_subplot(gs00[0])
    ax1.plot(f0, alpha = 0.6, label= f"$f0$") # rhat = {rhats[c,0]:.3})")
    ax1.plot(f0_param, alpha = 0.6, label= "$\sigma_{f_0} $")# + f"(rhat = {rhats[c,1]:.3})")
    ax1.set_ylabel('parameter value')
    ax1.set_xlabel('n')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.set_title('a)\nTrace and scatter plots of $f_0$ and $\sigma_{f_0}$')
    ax3 = fig.add_subplot(gs00[1])
    ax3.scatter(f0, f0_param, c=colors, alpha = 0.1)
    ax3.set_xlabel('$f_0$')
    ax3.set_ylabel('$\sigma_{f_0}$')
    ax3.legend(handles=handles, labels=labels, ncol=2, loc='upper left')


    # SUB2
    gs01 =  gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0,1], height_ratios=[0.2,0.7], hspace=0.5)
    ax2 = fig.add_subplot(gs01[0])
    ax2.plot(kappa, alpha = 0.6, label= f"$\kappa$")# (rhat = {rhats[c,2]:.3})")
    ax2.plot(sigma, alpha = 0.6, label= f"$\sigma$")# (rhat = {rhats[c,4]:.3})")
    ax2.set_xlabel('n')
    handles, labels = ax2.get_legend_handles_labels()
    ax2.set_title('b)\nTrace and scatter plots of $\kappa$ and $\sigma$')
    ax4 = fig.add_subplot(gs01[1])
    ax4.scatter(kappa, sigma, c=colors, alpha = 0.1)
    ax4.set_xlabel('$\kappa$')
    ax4.set_ylabel('$\sigma$')
    ax4.legend(handles=handles, labels=labels, ncol=2,loc='upper left')


    # SUB3
    gs02 =  gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0,2], height_ratios=[0.2,0.7], hspace=0.5)
    ax5 = fig.add_subplot(gs02[0])
    ax5.plot(sigma, alpha = 0.6, label= f"$\sigma$")# (rhat = {rhats[c,4]:.3})")
    ax5.plot(scale, alpha = 0.6, label= f"$\ell$")# (rhat = {rhats[c,3]:.3})")
    ax5.set_xlabel('n')
    handles, labels = ax5.get_legend_handles_labels()
    ax5.set_title('c)\nTrace and scatter plots of $\sigma$ and $\ell$')
    ax7 = fig.add_subplot(gs02[1])
    ax7.scatter(sigma,scale, c=colors, alpha = 0.1)
    ax7.set_xlabel('$\sigma$')
    ax7.set_ylabel('$\ell$')
    ax7.legend(handles=handles,labels=labels, ncol=2,loc='upper left')

    # SUB4
    gs10 =  gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1,0], height_ratios=[0.2,0.7], hspace=0.5)
    ax6 = fig.add_subplot(gs10[0])
    ax6.plot(alpha[0], alpha = 0.6, label= f"$\\alpha_0$")# (rhat = {rhats[c,5]:.3})")
    ax6.plot(alpha[1], alpha = 0.6, label= f"$\\alpha_1$")# (rhat = {rhats[c,6]:.3})")
    ax6.set_ylabel('parameter value')
    ax6.set_xlabel('n')
    handles, labels = ax6.get_legend_handles_labels()
    ax6.set_title('d)\nTrace and scatter plots of $\\alpha_0$ and $\\alpha_1$')
    ax8 = fig.add_subplot(gs10[1])
    ax8.scatter(alpha[0],alpha[1],c=colors, alpha = 0.1)
    ax8.set_xlabel('$\\alpha_0$')
    ax8.set_ylabel('$\\alpha_1$')
    ax8.legend(handles=handles,labels=labels, ncol=1,loc='upper left')

    # Sample plot
    ax9 = fig.add_subplot(gs[1,1:3])
    ax9.scatter(x_train,y_train, label='Training data', color='grey')
    # ax9.set_ylim(0,5)
    ax9.set_title('e) Function samples')

    if marked_centers != None:
        col = plt.get_cmap('tab10').colors
        custom_lines = [Line2D([0], [0], marker='o', linestyle='', color='grey', label=f'Training data')]
        clusters = []
        for k in range(len(marked_centers[0])):
            points = [i for i in np.where(np.abs(alpha[0] - marked_centers[0][k]) < tol)[0] if i in np.where(np.abs(alpha[1] - marked_centers[1][k]) < tol)[0]]
            clusters.append(points[:15])
            if len(points) > 0:
                custom_lines.append(Line2D([0], [0], color=col[k], lw=4, label=f'Cluster {k+1}'))

        for i,points in enumerate(clusters):
            ax3.scatter(f0[points], f0_param[points], alpha = 0.9)
            ax4.scatter(kappa[points], sigma[points], alpha = 0.9)
            ax7.scatter(sigma[points],scale[points], alpha = 0.9)
            ax8.scatter(alpha[0,points], alpha[1,points],alpha=0.9, label=str(i+1))
            ax9.plot(xpred, fpred[:,points], linewidth=1.5,alpha=0.5, label=str(i+1), color=col[i])
                
        ax9.legend(handles = custom_lines)
    else:
        for i in range(num_samples):
            if i % 100 == 0:
                ax9.plot(xpred, fpred[:,i], alpha=0.2, color=colors[i])
        ax9.legend()

    norm = matplotlib.colors.Normalize(vmin=0, vmax=10000)
    sm = matplotlib.cm.ScalarMappable(cmap='cool', norm=norm)
    sm.set_array([])  # Only needed for older versions of matplotlib
    cbar = fig.colorbar(sm, ax=ax9, orientation='horizontal', location='bottom')
    cbar.set_label('f) Colorbar indicating sample number')

r = 10
with open(path + 'model_selection_results.pkl' , 'rb') as file:
    model_dict = pickle.load(file)
m = model_dict['m']
L = model_dict['L']

c = 2
marked = [[0, -1.1, 1.1], [0,-1.5,1.5]]
convergece_plots(params,m,L, r, c, marked_centers=marked)
max_rhat = np.max(rhats[r,c])
plt.suptitle(f'Run {r+1}, chain {c+1}, '  +  '$\mathrm{max}\left(\hat{r}\\right) = $ ' + f'${max_rhat:.2f}$')
plt.tight_layout()
plt.savefig(path + 'exp1/convergence_plots_good_chain.png', bbox_inches='tight', dpi=500)
plt.show()

c = 0
convergece_plots(params,m,L, r, c)
max_rhat = np.max(rhats[r,c])
plt.suptitle(f'Run {r+1}, chain {c+1}, '  +  '$\mathrm{max}\left(\hat{r}\\right) = $ ' + f'${max_rhat:.2f}$')
plt.tight_layout()
plt.savefig(path + 'exp1/convergence_plots_bad_chain.png', bbox_inches='tight', dpi=500)
plt.show()
