#%%
import numpy as np
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import pickle
import matplotlib.gridspec as gridspec
from utils import benchmark_ushaped
from matplotlib.lines import Line2D
import os
path = '7_Discussion/' 
os.makedirs(path + 'exp2', exist_ok=True)


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
        return (xpL*xpL)/4 + (np.cos(lam_sum*xpL) - 1)/(2*lam_sum*lam_sum)

    else:
        return (1 - np.cos(lam_diff*xpL))/(2*lam_diff*lam_diff) + (np.cos(lam_sum*xpL) - 1)/(2*lam_sum*lam_sum)

def pred(m, L, theta_k, alpha, F0, f0, x_pred):
    kappa, scale = theta_k
    num_pred = len(x_pred)
    num_samples = len(kappa)

    PSI_array = np.zeros((m, m, num_pred))
    phi_array = np.zeros((m, num_pred))
    Lambda = np.zeros((m, num_samples))

    theta_k = np.stack([kappa,scale], axis = 0)
    f0pred = -f0*(x_pred[:,None] + L)

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
    fs += f0pred
    fs += F0

    return fs


exp = '6_Experiments/63_Ushaped_benchmark_functions'
function = 'sine'
b_function_index = 4
#1 good, 5 bad

path = exp +f'/HS_model/results/{function}/'

# Open the pickle file in binary read mode
with open(path + 'model_selection_results.pkl' , 'rb') as file:
    model_dict = pickle.load(file)

num_chains = 4
num_samples  = 10000
R = 2

seed = 12345*(b_function_index+1) + R
np.random.seed(seed)
x_train = np.random.uniform(-5, 5, 15)
y_train = benchmark_ushaped(b_function_index, x_train) + np.random.normal(0, 1, 15)

m = model_dict['m']
L = model_dict['L']

print(m)

params = np.load(path + 'params_and_rhat/10000_test_params_array.npy')
rhats = np.load(path + 'params_and_rhat/10000_test_rhat_array.npy')

print(rhats.shape)
#  ['F0', 'F0_param', 'f0', 'f0_param', 'kappa', 'scale', 'sigma', 'alpha']

xpred = np.linspace(-5,5,100)


def convergece_plots(params, xpred, m, L, r, c, marked_centers = None, tol=0.15):
    fig = plt.figure(figsize=(plotwidth, plotheight))
    gs = gridspec.GridSpec(3, 3, figure=fig) 

    chain_idx = np.arange(c,num_chains*num_samples, num_chains)
    print(params.shape)
    F0 = params[r,0,chain_idx]
    F0_param = params[r,1,chain_idx]
    f0 = params[r,2,chain_idx]
    f0_param = params[r,3,chain_idx]
    kappa = params[r,4,chain_idx]
    scale = params[r,5,chain_idx]
    sigma = params[r,6,chain_idx]
    alpha = params[r,7:,chain_idx].T
    fpred = pred(m=m, L=L, theta_k=[kappa,scale],alpha=alpha, f0=f0,F0=F0, x_pred=xpred)

    # SUB 1
    gs00 =  gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0,0], height_ratios=[0.2,0.7], hspace=0.5)
    ax1 = fig.add_subplot(gs00[0])
    ax1.plot(f0, alpha = 0.6, label= f"$f0$") # rhat = {rhats[c,0]:.3})")
    ax1.plot(F0, alpha = 0.6, label= "$F_0$")# + f"(rhat = {rhats[c,1]:.3})")
    ax1.set_ylabel('parameter value')
    ax1.set_xlabel('n')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.set_title('a)\nTrace and scatter plots of $f_0$ and $F_0$')
    ax2 = fig.add_subplot(gs00[1])
    ax2.scatter(f0, f0_param, c=colors, alpha = 0.1)
    ax2.set_xlabel('$f_0$')
    ax2.set_ylabel('$F_0$')
    ax2.legend(handles=handles, labels=labels, ncol=2, loc='upper left')


    # SUB2
    gs01 =  gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0,1], height_ratios=[0.2,0.7], hspace=0.5)
    ax3 = fig.add_subplot(gs01[0])
    ax3.plot(kappa, alpha = 0.6, label= f"$\kappa$")# (rhat = {rhats[c,2]:.3})")
    ax3.plot(sigma, alpha = 0.6, label= f"$\sigma$")# (rhat = {rhats[c,4]:.3})")
    ax3.set_xlabel('n')
    handles, labels = ax3.get_legend_handles_labels()
    ax3.set_title('b)\nTrace and scatter plots of $\kappa$ and $\sigma$')
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
    ax6 = fig.add_subplot(gs02[1])
    ax6.scatter(sigma,scale, c=colors, alpha = 0.1)
    ax6.set_xlabel('$\sigma$')
    ax6.set_ylabel('$\ell$')
    ax6.legend(handles=handles,labels=labels, ncol=2,loc='upper left')

    # SUB4
    gs10 =  gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1,0], height_ratios=[0.2,0.7], hspace=0.5)
    ax7 = fig.add_subplot(gs10[0])
    ax7.plot(alpha[0], alpha = 0.6, label= f"$\\alpha_0$")# (rhat = {rhats[c,5]:.3})")
    ax7.plot(alpha[1], alpha = 0.6, label= f"$\\alpha_1$")# (rhat = {rhats[c,6]:.3})")
    ax7.set_ylabel('parameter value')
    ax7.set_xlabel('n')
    handles, labels = ax7.get_legend_handles_labels()
    ax7.set_title('d)\nTrace and scatter plots of $\\alpha_0$ and $\\alpha_1$')
    ax8 = fig.add_subplot(gs10[1])
    ax8.scatter(alpha[0],alpha[1],c=colors, alpha = 0.1)
    ax8.set_xlabel('$\\alpha_0$')
    ax8.set_ylabel('$\\alpha_1$')
    ax8.legend(handles=handles,labels=labels, ncol=2,loc='upper left')

    # SUB5
    gs11 =  gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1,1], height_ratios=[0.2,0.7], hspace=0.5)
    ax9 = fig.add_subplot(gs11[0])
    ax9.plot(alpha[2], alpha = 0.6, label= f"$\\alpha_2$")# (rhat = {rhats[c,5]:.3})")
    ax9.plot(alpha[3], alpha = 0.6, label= f"$\\alpha_3$")# (rhat = {rhats[c,6]:.3})")
    ax9.set_ylabel('parameter value')
    ax9.set_xlabel('n')
    handles, labels = ax9.get_legend_handles_labels()
    ax9.set_title('d)\nTrace and scatter plots of $\\alpha_0$ and $\\alpha_1$')
    ax10 = fig.add_subplot(gs11[1])
    ax10.scatter(alpha[2],alpha[3],c=colors, alpha = 0.1)
    ax10.set_xlabel('$\\alpha_2$')
    ax10.set_ylabel('$\\alpha_3$')
    ax10.legend(handles=handles,labels=labels, ncol=2,loc='upper left')

    # SUB6
    gs12 =  gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1,2], height_ratios=[0.2,0.7], hspace=0.5)
    ax11 = fig.add_subplot(gs12[0])
    ax11.plot(alpha[4], alpha = 0.6, label= f"$\\alpha_4$")# (rhat = {rhats[c,5]:.3})")
    ax11.plot(alpha[0], alpha = 0.6, label= f"$\\alpha_0$")# (rhat = {rhats[c,6]:.3})")
    ax11.set_ylabel('parameter value')
    ax11.set_xlabel('n')
    handles, labels = ax11.get_legend_handles_labels()
    ax11.set_title('d)\nTrace and scatter plots of $\\alpha_4$ and $\\alpha_0$')
    ax12 = fig.add_subplot(gs12[1])
    ax12.scatter(alpha[4],alpha[0],c=colors, alpha = 0.1)
    ax12.set_xlabel('$\\alpha_0$')
    ax12.set_ylabel('$\\alpha_1$')
    ax12.legend(handles=handles,labels=labels, ncol=2,loc='upper left')

    gs20 =  gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2,0], height_ratios=[0.2,0.7], hspace=0.5)
    ax13 = fig.add_subplot(gs20[0])
    ax13.plot(f0_param, alpha = 0.6, label= "$\sigma_{f_0}$")# (rhat = {rhats[c,5]:.3})")
    ax13.plot(F0_param, alpha = 0.6, label= "$\sigma_{F_0}$")# (rhat = {rhats[c,6]:.3})")
    ax13.set_ylabel('parameter value')
    ax13.set_xlabel('n')
    handles, labels = ax13.get_legend_handles_labels()
    ax13.set_title('d)\nTrace and scatter plots of $\sigma_{f_0}$ and $\sigma_{F_0}$')
    ax14 = fig.add_subplot(gs20[1])
    ax14.scatter(f0_param,F0_param,c=colors, alpha = 0.1)
    ax14.set_xlabel("$\sigma_{f_0}$")
    ax14.legend(handles=handles,labels=labels, ncol=2,loc='upper left')

    # Sample plot
    ax15 = fig.add_subplot(gs[2,1:3])
    ax15.scatter(x_train,y_train, label='Training data', color='grey')
    # ax15.set_ylim(0,5)
    ax15.set_title('e) Function samples')

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
            ax1.scatter(f0[points], f0_param[points], alpha = 0.9)
            ax3.scatter(kappa[points], sigma[points], alpha = 0.9)
            ax5.scatter(sigma[points],scale[points], alpha = 0.9)
            ax7.scatter(alpha[0,points], alpha[1,points],alpha=0.9)
            ax9.scatter(alpha[2,points], alpha[3,points],alpha=0.9)
            ax11.scatter(alpha[4,points], alpha[0,points],alpha=0.9)
            ax13.scatter(f0_param[points], F0_param[points], alpha=0.9)
            ax15.plot(xpred, fpred[:,points], linewidth=1.5,alpha=0.5, label=str(i+1), color=col[i])
                
        ax15.legend(handles = custom_lines)
    else:
        for i in range(num_samples):
            if i % 100 == 0:
                ax15.plot(xpred, fpred[:,i], alpha=0.2, color=colors[i])
        ax15.legend()

    norm = matplotlib.colors.Normalize(vmin=0, vmax=10000)
    sm = matplotlib.cm.ScalarMappable(cmap='cool', norm=norm)
    sm.set_array([])  # Only needed for older versions of matplotlib
    cbar = fig.colorbar(sm, ax=ax15, orientation='horizontal', location='bottom')
    cbar.set_label('f) Colorbar indicating sample number')


# plt.plot(xpred,fpred_1)
# plt.savefig('7_Discussion/exp2/test.png')

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
plotheight = plotwidth*1.4
r = R
for c in range(4):
    convergece_plots(params, xpred, m, L, r, c,)
    max_rhat = np.max(rhats[r,c])
    plt.suptitle(f'Run {R}, chain {c+1}, '  +  '$\mathrm{max}\left(\hat{R}\\right) = $ ' + f'${max_rhat:.3f}$')
    plt.tight_layout()
    plt.show()
    plt.savefig(f'7_Discussion/exp2/convergence_plots_R={r}_c={c}.png', bbox_inches='tight', dpi = 500)


# convergece_plots(params_1, rhats_1, fpred_1, c)
# plt.suptitle(f'Chain {c+1}, Run 1')
# plt.tight_layout()
# plt.savefig('7_Discussion/exp2/convergence_plots_R1.png', bbox_inches='tight', dpi = 500)

# chain_idx = np.arange(c,num_chains*num_samples, num_chains)

# parameter_names = ['F0', 'F0_param', 'f0', 'f0_param', 'kappa', 'scale', 'sigma', 'alpha1', 'alpha2', 'alpha3', 'alpha4', 'alpha5']


# combs = [[0,1],
#          [0,2],
#          [2,3],
#          [4,5],
#          [4,6],
#          [5,6],
#          [7,8],
#          [7,9],
#          [7,10],
#          [7,11],
#          [8,9],
#          [8,10],
#          [8,11],
#          [9,10],
#          [9,11],
#          [10,11]
#          ]

# fig, ax = plt.subplots(len(combs),1, figsize=(10,len(combs)*5))

# for k,comb in enumerate(combs):
#     p1 = params_5[comb[0],chain_idx]
#     p1_name = parameter_names[comb[0]]
#     p2 = params_5[comb[1],chain_idx]
#     p2_name = parameter_names[comb[1]]
    
#     ax[k].scatter(p1,p2, color=colors)
#     ax[k].set_xlabel(p1_name)
#     ax[k].set_ylabel(p2_name)

# plt.savefig('7_Discussion/exp2/param_combs_5.png')


# for k,comb in enumerate(combs):
#     p1 = params_1[comb[0],chain_idx]
#     p1_name = parameter_names[comb[0]]
#     p2 = params_1[comb[1],chain_idx]
#     p2_name = parameter_names[comb[1]]
    
#     ax[k].scatter(p1,p2, color=colors)
#     ax[k].set_xlabel(p1_name)
#     ax[k].set_ylabel(p2_name)

# plt.savefig('7_Discussion/exp2/param_combs_1.png')