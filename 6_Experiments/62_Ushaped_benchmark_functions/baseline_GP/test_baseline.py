#%%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from matplotlib import pyplot as plt
from HSGP import HSGP_box_domain, GP_true, se_spectral_density_1D, se_spectral_density_grad, squared_exponential
from utils import compute_lpd, compute_RMSE, benchmark_ushaped
import pickle

b_functions = ['flat', 'skew', 'parabola', 'abs', 'sine', 'step']


seed_0 = 1234

path = '6_Experiments/63_Ushaped_benchmark_functions/baseline_GP/'

## SET PARAMETERS
num_experiments = 20
num_train = 15
num_test = 100
num_samples = 1000 # Number of posterior samples

x_pred = np.linspace( -5, 5, 100)
sigma_D = 1
theta_init = [1,1.5,1]   

for b_function_index, b_function in enumerate(b_functions):
    RMSE_array = np.zeros(num_experiments)
    lpd_array = np.zeros(num_experiments)

    for i in range(num_experiments):
        np.random.seed(seed_0 + i)
        # Generate data
        x_train = np.random.uniform(-5, 5, num_train)
        x_test = np.random.uniform(-5, 5, num_test)
        y_train = benchmark_ushaped(b_function_index, x_train) + np.random.normal(0, sigma_D, x_train.shape)
        y_test = benchmark_ushaped(b_function_index, x_test) + np.random.normal(0, sigma_D, x_test.shape)

        
        GP = GP_true(squared_exponential, theta=theta_init)
        sigma, kappa, scale = GP.optimize_hyperparameters(X=x_train[:,None], y=y_train[:,None],theta_init=theta_init)
        GP.fit(X=x_train[:,None], y=y_train[:,None])

        # Test
        f_test = GP.generate_samples(x_test[:,None], num_samples=num_samples).T
        lpd_array[i] = compute_lpd(y_test, f_test, GP.sigma)
        RMSE_array[i] = compute_RMSE(y_test, f_test)


    # Create results folder if it doesn't exist
    results_folder = path + 'results/' + b_function
    os.makedirs(results_folder, exist_ok=True)

    # Save train and test set ???
    np.save(f'{results_folder}/xtrain.npy',x_train)
    np.save(f'{results_folder}/ytrain.npy',y_train)
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
    test_results = np.zeros(4)
    test_results[0] = np.mean(RMSE_array)
    test_results[1] = np.std(RMSE_array)
    test_results[2] = np.mean(lpd_array)
    test_results[3] = np.std(lpd_array)

    np.save(f'{results_folder}/test_results.npy',test_results)

