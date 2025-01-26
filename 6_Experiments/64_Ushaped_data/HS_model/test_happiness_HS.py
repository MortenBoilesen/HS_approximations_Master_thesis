import stan
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pickle
from utils import compute_RMSE, compute_lpd, compute_rhat


def run_simulation(simulation_code, simulation_data, y_test, parameter_names, seed = 0 ):
    # Build model
    posterior = stan.build(simulation_code, data=simulation_data, random_seed= seed)

    fit = posterior.sample(num_chains=args.num_chains, num_samples=args.num_samples, num_warmup=args.num_warmup)
        
    param_list = []
    for p in parameter_names:
        param = fit[p]
        for j in range(len(param)):
            param_list.append(param[j])

    params = np.stack(param_list)
    rs = compute_rhat(parameter_array=params, num_chains=args.num_chains, num_samples=args.num_samples)
        
    rhat_test = np.prod((rs < 1.1), dtype=bool)
    ftrain = fit["f_test"]
    fpred = fit["f_pred"]
    sigma = fit["sigma"]
    lpd = compute_lpd(y_test, ftrain, sigma)
    RMSE = compute_RMSE(y_test, ftrain)

    return RMSE, lpd, rhat_test, fpred, sigma, params, rs


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--R', type=int, default=2, help='What run (default: %(default)s)')
    parser.add_argument('--num_chains', type=int, default=3, help='number of chains in HMC sampler (default: %(default)s)')
    parser.add_argument('--num_samples', type=int, default=3000, help='number of samples in each chain (default: %(default)s)')
    parser.add_argument('--num_warmup', type=int, default=1000, help='number of samples in warm up (default: %(default)s)')
    parser.add_argument('--fraction', type=float, default=1.0, help='fraction of data to use measured in eighths (default: %(default)s)')

    args = parser.parse_args()

    parameter_names = ['F0', 'F0_param', 'f0', 'f0_param', 'kappa', 'scale', 'sigma', 'alpha']
    
    # Paths to data and model
    path = f'6_Experiments/64_Ushaped_data/Runs/Run_{args.R}/'
    datapath = path + '/00data/'
    resultpath = path + f'HS_model/{args.fraction}/'
    params_rhat_path = resultpath + 'params_and_rhat/'

    os.makedirs(resultpath, exist_ok = True)
    os.makedirs(params_rhat_path, exist_ok = True)

    # Load code
    with open("6_Experiments/64_Ushaped_data/HS_model/HSGP_concave_pred.stan", "r") as f:
        simulation_code = f.read()

    # Load best parameters
    # with open(resultpath+'model_selection_results.pkl', 'rb') as file:
    #     best_configuration = pickle.load(file)


    # Load and crop training data
    x_train = np.load(datapath+f'x_train_shuffled.npy')
    y_train = np.load(datapath+f'y_train_shuffled.npy')

    num_total = len(x_train) 
    num_train = int(num_total * args.fraction / 8)

    x_train = x_train[:num_train]
    y_train = y_train[:num_train]

    # Load test data
    x_test = np.load(datapath+f'x_test.npy')
    y_test = np.load(datapath+f'y_test.npy')
    num_test = len(x_test)

    num_pred = 100
    x_pred = np.linspace(-1.8,1.8,num_pred)

    # Load best parameters
    with open(resultpath+'model_selection_results.pkl', 'rb') as file:
        best_configuration = pickle.load(file)



    if best_configuration["L"] == 2:
        F_0_mean = 2
    elif best_configuration["L"] == 3:
        F_0_mean = 4
    elif best_configuration["L"] == 5:
        F_0_mean = 7.5

    simulation_data = {
        "num_test": num_test,
        "xtest": x_test,
        "ytest": y_test,

        "num_train": num_train,
        "xtrain": x_train,
        "ytrain": y_train,

        "num_pred": num_pred,
        "xpred": x_pred,

        "F0_mean": F_0_mean,

        "jitter": 1e-8,
        "num_basis_functions": best_configuration["m"],
        "L": best_configuration["L"]
    }

    best_configuration_results = np.zeros(2)
    
    RMSE, lpd, rhat_test, fpred, sigma, params, rs = run_simulation(simulation_code, simulation_data, y_test, parameter_names, seed = 1234)

    best_configuration_results[0] = RMSE 
    best_configuration_results[1] = lpd 

   # Save the best results
    np.save(resultpath+'test_results.npy', (best_configuration_results))
    np.save(resultpath+'test_fpred.npy', (fpred))
    np.save(resultpath+'test_sigma.npy', (sigma))

    np.save(params_rhat_path+'test_params.npy', (params))
    np.save(params_rhat_path+'test_rhat.npy', (rs))
    