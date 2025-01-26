import stan
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pickle
from utils import compute_RMSE, compute_lpd, compute_rhat


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--R', type=int, default=2, help='What run (default: %(default)s)')
    parser.add_argument('--num_chains', type=int, default=3, help='number of chains in HMC sampler (default: %(default)s)')
    parser.add_argument('--num_samples', type=int, default=500, help='number of samples in each chain (default: %(default)s)')
    parser.add_argument('--num_warmup', type=int, default=500, help='number of samples in warm up (default: %(default)s)')
    parser.add_argument('--fraction', type=float, default=1.0, help='fraction of data to use measured in eigtths (default: %(default)s)')
    parser.add_argument('--nu', type=float, default=1e2, help='controls the strictness of monotonicity information in the virtual points (default: %(default)s)')

    args = parser.parse_args()

    parameter_names = ['scale', 'kappa', 'sigma', 'eta', 'f0']    

    # Paths to data and model
    path = f'6_Experiments/64_Ushaped_data/Runs/Run_{args.R}/'
    datapath = path + '/00data/'
    resultpath = path + f'VP_model/{args.fraction}/'
    params_rhat_path = resultpath + 'params_and_rhat/'

    os.makedirs(resultpath, exist_ok = True)
    os.makedirs(params_rhat_path, exist_ok = True)

    # Load code
    with open("6_Experiments/64_Ushaped_data/VP_model/gp_convex_gaussian_virtual_pred.stan", "r") as f:
        simulation_code = f.read()

    # Load best parameters
    with open(resultpath+'model_selection_results.pkl', 'rb') as file:
        best_configuration = pickle.load(file)

    num_virtual = best_configuration["num_virtual"]

    # Load and crop training data
    x_train = np.load(datapath+f'x_train_shuffled.npy')
    y_train = np.load(datapath+f'y_train_shuffled.npy')

    num_total = len(x_train) 
    num_train = int(num_total * args.fraction / 8)

    x_train = x_train[:num_train]
    y_train = y_train[:num_train]

    x_test = np.load(datapath+f'x_test.npy')
    y_test = np.load(datapath+f'y_test.npy')
    num_test = len(x_test)

    num_pred = 100
    x_pred = np.linspace(-1.8,1.8, num_pred)

    xvirtual = np.linspace(-1.8, 1.8, num_virtual)
    yvirtual = np.zeros(num_virtual, dtype=int)

    # Load best parameters
    with open(resultpath+'model_selection_results.pkl', 'rb') as file:
        best_configuration = pickle.load(file)

    simulation_data = {
        "num_test": num_test,
        "xtest": x_test,
        "ytest": y_test,

        "num_train": num_train,
        "xtrain": x_train,
        "ytrain": y_train,

        "num_virtual": num_virtual,
        "xvirtual": xvirtual,
        "yvirtual": yvirtual,
        "nu": args.nu,

        "num_pred": num_pred,
        "xpred": x_pred,

        "jitter": 1e-8
    }

    # Build model
    posterior = stan.build(simulation_code, data=simulation_data, random_seed = 1234)

    fit = posterior.sample(num_chains=args.num_chains, num_samples=args.num_samples, num_warmup=args.num_warmup)
        
    param_list = []
    for p in parameter_names:
        param = fit[p]
        for j in range(len(param)):
            param_list.append(param[j])

    params = np.stack(param_list)
    rs = compute_rhat(parameter_array=params, num_chains=args.num_chains, num_samples=args.num_samples)
    rhat_test = np.prod((rs < 1.1), dtype=bool)

    f0 = fit["f0"]
    ftest = f0 + fit["f_test"]
    fpred = f0 + fit["f_pred"]
    sigma = fit["sigma"]
    lpd = compute_lpd(y_test, ftest, sigma)
    RMSE = compute_RMSE(y_test, ftest)


    best_configuration_results = np.zeros(2)
    best_configuration_results[0] = RMSE 
    best_configuration_results[1] = lpd 

   # Save the best results
    np.save(resultpath+'test_results.npy', (best_configuration_results))
    np.save(resultpath+'test_fpred.npy', (fpred)) ## Effective sample size?
    np.save(resultpath+'test_sigma.npy', (sigma))

    np.save(params_rhat_path+'test_params.npy', (params))
    np.save(params_rhat_path+'test_rhat.npy', (rs))
    