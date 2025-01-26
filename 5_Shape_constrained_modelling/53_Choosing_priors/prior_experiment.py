import stan
from matplotlib import pyplot as plt
import numpy as np
import time
import scipy.stats as sp
import scipy.special as ss
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import compute_lpd

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='', help='relative path to data located in F_prior_experiments/data folder.')
    parser.add_argument('--f0_prior', type=str, default='normal', choices=['standardnormal','normal','gamma', 'lognormal'], help='Prior on f0 (default: %(default)s)')
    parser.add_argument('--F0_prior', type=str, default='normal', choices=['standardnormal', 'normal', 'studentt'], help='Prior on F0 (default: %(default)s)')
    parser.add_argument('--F0_prior_mean', action='store_true')
    parser.add_argument('--num_chains', type=int, default=3, help='Number of chains in HMC sampler (default: %(default)d).')
    parser.add_argument('--num_samples', type=int, default=3000, help='Number of samples per chain in HMC sampler (default: %(default)d).')
    parser.add_argument('--num_warmup', type=int, default=1000, help='Number of warmup samples per chain in HMC sampler (default: %(default)d).')
    parser.add_argument('--num_basis_functions', type=int, default=10, help='Number of basis functions to use (default: %(default)d).')
    parser.add_argument('--L', type=int, default=5, help='Value of L (default: %(default)d)')
    
    args = parser.parse_args()
    print('f0 prior: '+args.f0_prior+', F0 prior: '+args.F0_prior)
    print(args.F0_prior_mean)

    path = 'experiment_2_F_prior/'
    
    with open(path + 'prior_experiment.stan', "r") as f:
        ushaped_simulation_code = f.read()

    xtest = np.load(path+'data/xtest.npy')
    ytest = np.load(path+'data/ytest.npy')

    xtrain = np.load(path+'data/xtrain.npy')
    ytrain = np.load(path+'data/ytrain.npy')

    num_test = len(xtest)
    num_train = len(xtrain)


    if args.F0_prior_mean:
        sort_idx = np.argsort(xtrain)
        index = sort_idx[:(num_train // 4)]
        print(index)
        # Create the subset
        x_train_subset = xtrain[index]
        y_train_subset = ytrain[index]

        X_b = np.c_[np.ones((x_train_subset.shape[0], 1)), x_train_subset]

        # Calculate the optimal theta using the normal equation
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train_subset)
        print(theta_best)
        F0_prior_mean = theta_best[0] + theta_best[1] * (-args.L)
    else:
        F0_prior_mean = 0

    print(F0_prior_mean)

    num_pred = 100
    xpred = np.linspace(-2.5,2.5, num_pred)

    ushaped_simulation_data = {
                    "num_test": num_test,
                    "xtest": xtest,
                    "ytest": ytest,
                    
                    "num_train": num_train,
                    "xtrain": xtrain,
                    "ytrain": ytrain,

                    "num_pred": num_pred,
                    "xpred": xpred,

                    "jitter": 1e-8,

                    "F0_prior_mean": F0_prior_mean,

                    "num_basis_functions": args.num_basis_functions,
                    "L": args.L,
                    }
    
    if args.f0_prior == 'normal':
        ushaped_simulation_data['f0_prior'] = 0  
    elif args.f0_prior == 'gamma':
        ushaped_simulation_data['f0_prior'] = 1       
    elif args.f0_prior == 'lognormal': 
        ushaped_simulation_data['f0_prior'] = 2
    else:
        ushaped_simulation_data['f0_prior'] = 3
    

    if args.F0_prior == 'normal':
        ushaped_simulation_data['F0_prior'] = 0
    elif args.F0_prior == 'studentt': 
        ushaped_simulation_data['F0_prior'] = 1
    else: 
        ushaped_simulation_data['F0_prior'] = 2

    posterior = stan.build(ushaped_simulation_code, data=ushaped_simulation_data, random_seed=1234)

    num_chains = args.num_chains
    num_samples = args.num_samples

    start_time = time.time()
    fit = posterior.sample(num_chains=num_chains, num_samples=num_samples, num_warmup=args.num_warmup)
    end_time = time.time()

    print('sampling done')

    execution_time = end_time - start_time
    print(f"Fit execution time: { int(execution_time / 60) } minutes and {int(execution_time % 60)} seconds")

    ftrain = fit["f_train"]
    ftest = fit["f_test"]
    fpred = fit["f_pred"]
    sigma = fit["sigma"]
    lpd = compute_lpd(y_test=ytest, f_test=ftest, sigma=sigma)

    if args.F0_prior_mean:
        result_path = path+'/results/'+args.f0_prior+'_'+args.F0_prior + '_with_mean/'
    else:
        result_path = path+'/results/'+args.f0_prior+'_'+args.F0_prior + '/'

    os.makedirs(result_path, exist_ok=True)

    np.save(result_path+f'ftrain.npy', ftrain)
    np.save(result_path+f'ftest.npy', ftest)
    np.save(result_path+f'fpred.npy', fpred)
    np.save(result_path+f'lpd.npy', lpd)
    np.save(result_path+f'sigma.npy', sigma)

    print('Saving data done')