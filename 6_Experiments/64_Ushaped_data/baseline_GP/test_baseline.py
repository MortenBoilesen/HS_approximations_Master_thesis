#%%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from HSGP import GP_true, squared_exponential
from utils import compute_lpd, compute_RMSE

# fractions = [8/16, 8/14, 8/10, 1,2,3,4,6,8]
fractions = [0.5, 0.7272727272727273, 0.8888888888888888, 1, 2, 4, 8]
## SET PARAMETERS
num_experiments = 20
num_samples = 1000 # Number of posterior samples

x_pred = np.linspace( -1.8, 1.8, 100)

for R in range(1,num_experiments+1):
    try:
        x_train_shuffled = np.load(f'6_Experiments/64_Ushaped_data/Runs/Run_{R}/00data/x_train_shuffled.npy')
        y_train_shuffled = np.load(f'6_Experiments/64_Ushaped_data/Runs/Run_{R}/00data/y_train_shuffled.npy')

        x_test = np.load(f'6_Experiments/64_Ushaped_data/Runs/Run_{R}/00data/x_test.npy')
        y_test = np.load(f'6_Experiments/64_Ushaped_data/Runs/Run_{R}/00data/y_test.npy')


    except:
        np.random.seed(100 + R)

        x_samples = np.load('6_Experiments/64_Ushaped_data/all_data/x_samples.npy')
        y_samples = np.load('6_Experiments/64_Ushaped_data/all_data/y_samples.npy')
            
        total_samples = len(x_samples) 
        ratio = 0.25
        num_test = int(ratio*total_samples)
        num_train = total_samples - num_test

        train_idx = np.random.choice(np.arange(total_samples,dtype=int), num_train, replace=False)
        test_idx = np.setdiff1d(np.arange(total_samples), train_idx)

        x_train = x_samples[train_idx]
        y_train = y_samples[train_idx]
        x_test = x_samples[test_idx]
        y_test = y_samples[test_idx]

        np.random.shuffle(train_idx)
        x_train_shuffled = x_samples[train_idx]
        y_train_shuffled = y_samples[train_idx]

        os.makedirs(f'6_Experiments/64_Ushaped_data/Runs/Run_{R}/00data/', exist_ok=True)

        np.save(f'6_Experiments/64_Ushaped_data/Runs/Run_{R}/00data/x_train.npy', x_train)
        np.save(f'6_Experiments/64_Ushaped_data/Runs/Run_{R}/00data/y_train.npy', y_train)
        np.save(f'6_Experiments/64_Ushaped_data/Runs/Run_{R}/00data/x_train_shuffled.npy', x_train_shuffled)
        np.save(f'6_Experiments/64_Ushaped_data/Runs/Run_{R}/00data/y_train_shuffled.npy', y_train_shuffled)
        np.save(f'6_Experiments/64_Ushaped_data/Runs/Run_{R}/00data/x_test.npy', x_test)
        np.save(f'6_Experiments/64_Ushaped_data/Runs/Run_{R}/00data/y_test.npy', y_test)


    for fraction in fractions:
        RMSE_array = np.zeros(num_experiments)
        lpd_array = np.zeros(num_experiments)

        num_train = int(len(x_train_shuffled) * fraction // 8)

        x_train = x_train_shuffled[:num_train]
        y_train = y_train_shuffled[:num_train]

        sigma_init = np.random.uniform(0.5,1.5)
        kappa_init = np.random.uniform(0.5,1.5)
        scale_init = np.random.uniform(0.5,1.5)

        theta_init = [sigma_init, kappa_init, scale_init]        

        GP = GP_true(squared_exponential, theta=theta_init)

        success = 0
        while type(success) == int:
            sigma_init = np.random.uniform(0.5,2)
            kappa_init = np.random.uniform(0.5,2)
            scale_init = np.random.uniform(0.5,2)
            theta_init = [sigma_init, kappa_init, scale_init]        
            success = GP.optimize_hyperparameters(X=x_train[:,None], y=y_train[:,None],theta_init=theta_init)
        print(success) 

        GP.fit(X=x_train[:,None], y=y_train[:,None])

        # Test
        f_test = GP.generate_samples(x_test[:,None], num_samples=num_samples).T
        lpd = compute_lpd(y_test, f_test, GP.sigma)
        RMSE = compute_RMSE(y_test, f_test)

        # Create results folder if it doesn't exist
        resultpath = f'6_Experiments/64_Ushaped_data/Runs/Run_{R}/baseline_GP/{str(fraction)}/'
        os.makedirs(resultpath, exist_ok=True)

        # Predict
        f_pred, mu, Sigma = GP.generate_samples(x_pred[:,None], num_samples=num_samples, return_distribution=True)
        f_pred = f_pred.T
        std = np.sqrt(np.diag(Sigma) + GP.sigma2)
        np.save(resultpath+'/test_fpred.npy',f_pred)
        np.save(resultpath+'/test_mu.npy',mu)
        np.save(resultpath+'/test_std.npy',std)

        # Compute test results
        test_results = np.array([RMSE, lpd])

        np.save(resultpath+'/test_results.npy',test_results)

