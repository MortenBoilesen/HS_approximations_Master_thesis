import stan
import numpy as np
import time
import sys
import os
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils import compute_RMSE, compute_lpd, compute_rhat


def run_simulation(m, L, simulation_code, parameter_names, x_data, y_data):
    rhat_test = np.zeros(args.num_experiments, dtype=bool)
    lpd_array = np.zeros(args.num_experiments)
    RMSE_array = np.zeros(args.num_experiments)

    for i in range(0,args.num_experiments):
        np.random.seed(i)
        # Generate data
        x_train = x_data[i*2:36 + i*2]
        x_test = x_data[36 + i*2:42 + i*2]
        y_train =  y_data[i*2:36 + i*2]
        y_test =  y_data[36 + i*2:42 + i*2]
        num_train = len(x_train)
        num_test = len(x_test)


        simulation_data = { 
            "num_test": num_test,
            "xtest": x_test,
            "ytest": y_test,

            "num_train": num_train,
            "xtrain": x_train,
            "ytrain": y_train,

            "jitter": 1e-8,
            "num_basis_functions": m,
            "L": L
        }

    # Build model
        posterior = stan.build(simulation_code, data=simulation_data, random_seed=i)

        fit = posterior.sample(num_chains=args.num_chains, num_samples=args.num_samples, num_warmup=args.num_warmup)
        
        param_list = []
        for p in parameter_names:
            param = fit[p]
            for j in range(len(param)):
                param_list.append(param[j])

        params = np.stack(param_list)
        rs = compute_rhat(parameter_array=params, num_chains=args.num_chains, num_samples=args.num_samples)
        rhat_test[i-1] = np.prod((rs < 1.1), dtype=bool)
        ftest = fit["f_test"]
        sigma = fit["sigma"]
        lpd_array[i-1] = compute_lpd(y_test, ftest, sigma)
        RMSE_array[i-1] = compute_RMSE(y_test, ftest)

    lpd_mean = np.mean(lpd_array)
    RMSE_mean = np.mean(RMSE_array)
    lpd_std = np.std(lpd_array)
    RMSE_std = np.std(RMSE_array)

    return RMSE_mean, RMSE_std, lpd_mean, lpd_std, rhat_test, RMSE_array, lpd_array


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_chains', type=int, default=3, help='number of chains in HMC sampler (default: %(default)s)')
    parser.add_argument('--num_samples', type=int, default=300, help='number of samples in each chain (default: %(default)s)')
    parser.add_argument('--num_warmup', type=int, default=300, help='number of samples in warm up (default: %(default)s)')
    parser.add_argument('--num_experiments', type=int, default=5, help='number of experiments (default: %(default)s)')
  
    args = parser.parse_args()

    print(f'number of runs: {args.num_experiments}, number of samples: {args.num_samples}, number of warmup: {args.num_warmup}.')
    # Paths to data and model
    path = '5_Shape_constrained_modelling/53_Relaxing_convexity/HS_model/'
    datapath = '5_Shape_constrained_modelling/53_Relaxing_convexity/00data/'
    resultpath = path + 'results/'
    samplepath = resultpath + 'experiment_samples/'

    os.makedirs(resultpath, exist_ok = True)
    os.makedirs(samplepath, exist_ok = True)

    parameter_names = ['f0', 'f0_param', 'kappa', 'scale', 'sigma']
    
    # Load code
    with open(path + "HSGP_monotonic.stan", "r") as f:
        simulation_code = f.read()

    m_list = [20]# 15,  25, 30]
    L_list = [3, 5] #[10, 20, 30]
    best_m = 100
    best_L = 100
    best_RMSE = 100

    # with open(f'{path}/results/model_selection_results.txt', 'w') as file:
    #     # file.write(f"Forecasting model selection\n")
    
    # Load data
    x_data = np.load(datapath + 'Fertility_rate/x_train.npy')
    y_data = np.load(datapath + 'Fertility_rate/y_train.npy')

    for m in m_list:
        for L in L_list:
            print(f'm = {m} and L={L}')
            result = run_simulation(m, L, simulation_code, parameter_names, x_data, y_data)
            print(f"Experiments done for m = {m} and L = {L}.")

            RMSE_mean, RMSE_std, lpd_mean, lpd_std, rhat_test, RMSE_array, lpd_array = result
            with open(f'{path}/results/model_selection_results.txt', 'a') as file:
                file.write(f"m: {m}, L: {L}, RMSE: {RMSE_mean:.3f}, RMSE_std: {RMSE_std:.3f}, lpd: {lpd_mean:.3f}, lpd_std: {lpd_std:.3f}, convergence test: {rhat_test} \n")
            
            np.save(samplepath + f'RMSE_m_{m}_L_{L}.npy', RMSE_array)
            np.save(samplepath + f'lpd_m_{m}_L_{L}.npy', lpd_array)

            if RMSE_mean < best_RMSE:
                best_RMSE = RMSE_mean
                best_m = m
                best_L = L

    # Save the best results
    best_configuration = {'m': best_m, 'L': best_L} 
    print(f'best configuration is m = {best_m} and L = {best_L}')
    with open(f'{path}results/model_selection_results.pkl', 'wb') as file:
        pickle.dump(best_configuration, file)