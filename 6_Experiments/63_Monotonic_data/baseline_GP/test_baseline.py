#%%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from matplotlib import pyplot as plt
from HSGP import HSGP_box_domain, GP_true, se_spectral_density_1D, se_spectral_density_grad, squared_exponential
from utils import compute_lpd, compute_RMSE, benchmark_functions

seed_0 = 1234
path = '5_Shape_constrained_modelling/53_Relaxing_convexity/baseline_GP/'
datapath = '5_Shape_constrained_modelling/53_Relaxing_convexity/00data/'


## SET PARAMETERS
num_samples = 1000 # Number of posterior samples

sigma_D = 1
theta_init = [1,1.5,1]   

x_train = np.load(datapath + 'Fertility_rate/x_train.npy')
y_train = np.load(datapath + 'Fertility_rate/y_train.npy')
x_test = np.load(datapath + 'Fertility_rate/x_test.npy')
y_test = np.load(datapath + 'Fertility_rate/y_test.npy')
x_pred = np.linspace(np.min(x_train), np.max(x_test), 100)

GP = GP_true(squared_exponential, theta=theta_init)
_ = GP.optimize_hyperparameters(X=x_train[:,None], y=y_train[:,None],theta_init=theta_init)
GP.fit(X=x_train[:,None], y=y_train[:,None])

# Test
f_test = GP.generate_samples(x_test[:,None], num_samples=num_samples).T
lpd_array = compute_lpd(y_test, f_test, GP.sigma)
RMSE_array = compute_RMSE(y_test, f_test)

# Create results folder if it doesn't exist
results_folder = path + 'results/'
os.makedirs(results_folder, exist_ok=True)

# Save train and test set ???
np.save(f'{results_folder}/xtest.npy',x_test)
np.save(f'{results_folder}/ytest.npy',y_test)
np.save(f'{results_folder}/ftest.npy',f_test)

# Predict
f_pred, mu, Sigma = GP.generate_samples(x_pred[:,None], num_samples=num_samples, return_distribution=True)
f_pred = f_pred.T
std = np.sqrt(np.diag(Sigma) + GP.sigma2)
np.save(f'{results_folder}/test_fpred.npy',f_pred)
np.save(f'{results_folder}/test_mu.npy',mu)
np.save(f'{results_folder}/test_sigma.npy',GP.sigma)
np.save(f'{results_folder}/test_std.npy',std)

# Compute test results
test_results = np.zeros(2)
test_results[0] = RMSE_array
test_results[1] = lpd_array

np.save(f'{results_folder}/test_results.npy',test_results)

# %%
