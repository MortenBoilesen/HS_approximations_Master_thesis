import stan
from matplotlib import pyplot as plt
import numpy as np
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import compute_RMSE, compute_lpd


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_chains', type=int, default=3, help='number of chains in HMC sampler (default: %(default)s)')
    parser.add_argument('--num_samples', type=int, default=1000, help='number of samples in each chain (default: %(default)s)')
    parser.add_argument('--m', type=float, default=10, help='number of basis function. (default: %(default)s)')
    parser.add_argument('--L', type=float, default=5, help='size of domain, [-L,L] (default: %(default)s)')
    parser.add_argument('--num_warmup', type=int, default=1000 , help='number of samples in warm up (default: %(default)s)')

    args = parser.parse_args()

    ## Paths to data and model
    path = '5_Shape_constrained_modelling/53_Relaxing_convexity/HS_model/'
    datapath = '5_Shape_constrained_modelling/53_Relaxing_convexity/00data/earth_temperature_data/'

    ## load code 
    
    with open(path+"HSGP_monotonic.stan", "r") as f:
        simulation_code = f.read()


    x_train = np.load(datapath+'x_train.npy')
    y_train = np.load(datapath+'y_train.npy')

    num_train = len(x_train)

    x_test = np.load(datapath+'x_test.npy')
    y_test = np.load(datapath+'y_test.npy')

    num_test = len(x_test)

    num_pred = 100
    x_pred = np.linspace(-1.8,1.8, num_pred)

    print(f"Number of train: {num_train}, Number of test: {num_test}, Number of prediction: {num_pred}")

    simulation_data = {
                    "num_test": num_test,
                    "xtest": x_test,
                    "ytest": y_test,
                    
                    "num_train": num_train,
                    "xtrain": x_train,
                    "ytrain": y_train,
                    
                    "num_pred": num_pred,
                    "xpred": x_pred,

                    "num_basis_functions": args.m,
                    "L": args.L,

                    "jitter": 1e-8,
                    }

    # Build model
    posterior = stan.build(simulation_code, data=simulation_data, random_seed=1)

    start_time = time.time()
    fit = posterior.sample(num_chains=args.num_chains, num_samples=args.num_samples, num_warmup=args.num_warmup)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Sampling done in {int(elapsed_time // 60)} minuts and {int(elapsed_time % 60)} seconds.")


    # Save samples
    ftrain = fit["f_train"]
    ftest = fit["f_test"]
    fpred = fit["f_pred"]
    sigma = fit["sigma"]

    lpd = compute_lpd(y_test, ftest, sigma)
    rmse = compute_RMSE(y_test, ftest)

    np.save(path+f'results/ftrain.npy', ftrain)
    np.save(path+f'results/ftest.npy', ftest)
    np.save(path+f'results/fpred.npy', fpred)
    np.save(path+f'results/sigma.npy', sigma)
    np.save(path+f'results/lpd.npy', lpd)
    np.save(path+f'results/rmse.npy', rmse)
    

    print("Saving done")