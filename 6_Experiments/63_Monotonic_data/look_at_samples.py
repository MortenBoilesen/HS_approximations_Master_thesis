import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from utils import plot_with_samples

from matplotlib import pyplot as plt
plt.rcParams.update({
    'font.size': 16.0,
    'axes.titlesize': 'medium'
})

#     parameter_names = ['f0', 'f0_param', 'kappa', 'scale', 'sigma', 'alpha']

params = np.load('5_Shape_constrained_modelling/53_Relaxing_convexity/HS_model/results/params_and_rhat/10000_test_params.npy')

xtrain = np.load('5_Shape_constrained_modelling/53_Relaxing_convexity/00data/Fertility_rate/x_train.npy')
ytrain = np.load('5_Shape_constrained_modelling/53_Relaxing_convexity/00data/Fertility_rate/y_train.npy')
xtest = np.load('5_Shape_constrained_modelling/53_Relaxing_convexity/00data/Fertility_rate/x_test.npy')
ytest = np.load('5_Shape_constrained_modelling/53_Relaxing_convexity/00data/Fertility_rate/y_test.npy')

xpred_true = np.linspace(np.min(xtrain), np.max(xtest), 100)
f_true = np.load('5_Shape_constrained_modelling/53_Relaxing_convexity/HS_model/results/10000_test_fpred.npy')


f0 = params[0,:]
kappa = params[2,:]
scale = params[3,:]
sigma = params[4,:]
alpha = params[5:,:]

print(scale.min(), scale.mean(), np.median(scale), scale.max())

print(alpha.shape)

num_pred = 100
L = 10
m = 5
# xpred = np.linspace(-L-1,L+1, num_pred)
print(min(xtrain))
print(max(xtrain))
xpred = np.linspace(-11,11,num_pred)

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
print(alpha.shape)

fs = np.zeros((num_pred, 40000))
gp = np.zeros((num_pred, 40000))

for i in range(m):
    gp += alpha[i]*phi_array[i][:,None]
    for j in range(m):
        fs += alpha[i]*alpha[j]*PSI_array[i,j][:,None]

fs += f0
fs = -fs
# fs = fs.T


# plt.plot(xpred, fs[:,:10], color='tab:green', alpha=0.2)
textwidth = 5.9
fig, ax = plt.subplots(1,1,figsize=(textwidth, 2/3*textwidth))
plot_with_samples(ax,xpred,fs,sigma,num_samples=100)
# ax.plot(xpred, fs[:,:100], color='tab:red', alpha=0.2)
# ax.plot(xpred_true, f_true[:,:100], color='tab:blue')    

ax.scatter(xtrain,ytrain,label='train')
ax.scatter(xtest,ytest, label='test')
ax.legend()

# ax[1].plot(xpred, gp[:,:100])
# ax[1].scatter(xtrain,ytrain,label='train')
# ax[1].scatter(xtest,ytest, label='test')

plt.tight_layout()
plt.savefig('temp.png', dpi=200, bbox_inches='tight')

