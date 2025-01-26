import stan
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pickle
from utils import compute_RMSE, compute_lpd, benchmark_ushaped, compute_rhat


def run_simulation(m, L, simulation_code, parameter_names, seed_0 = 0 ):
    rhat_test = np.zeros(args.num_experiments, dtype=bool)
    lpd_array = np.zeros(args.num_experiments)
    RMSE_array = np.zeros(args.num_experiments)
    
    for i in range(args.num_experiments):
        np.random.seed(i)
        # Generate data
        x_train = np.random.uniform(-5, 5, args.num_train)
        x_test = np.random.uniform(-5, 5, args.num_test)
        y_train = benchmark_ushaped(args.b_function_index, x_train) + np.random.normal(0, args.std, args.num_train)
        y_test = benchmark_ushaped(args.b_function_index, x_test) + np.random.normal(0, args.std, args.num_test)

        simulation_data = {
            "num_test": args.num_test,
            "xtest": x_test,
            "ytest": y_test,

            "num_train": args.num_train,
            "xtrain": x_train,
            "ytrain": y_train,

            "jitter": 1e-8,
            "num_basis_functions": m,
            "L": L
        }

        # Build model
        posterior = stan.build(simulation_code, data=simulation_data, random_seed=i + seed_0)

        fit = posterior.sample(num_chains=args.num_chains, num_samples=args.num_samples, num_warmup=args.num_warmup)
        
        param_list = []
        for p in parameter_names:
            param = fit[p]
            for j in range(len(param)):
                param_list.append(param[j])

        params = np.stack(param_list)
        rs = compute_rhat(parameter_array=params, num_chains=args.num_chains, num_samples=args.num_samples)
        
        rhat_test[i] = np.prod((rs < 1.1), dtype=bool)
        ftest = fit["f_test"]
        sigma = fit["sigma"]
        lpd_array[i] = compute_lpd(y_test, ftest, sigma)
        RMSE_array[i] = compute_RMSE(y_test, ftest)

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
    parser.add_argument('--num_samples', type=int, default=500, help='number of samples in each chain (default: %(default)s)')
    parser.add_argument('--num_warmup', type=int, default=500, help='number of samples in warm up (default: %(default)s)')
    parser.add_argument('--num_experiments', type=int, default=20, help='number of experiments (default: %(default)s)')
    parser.add_argument('--num_train', type=int, default=15, help='number of training points (default: %(default)s)')
    parser.add_argument('--num_test', type=int, default=100, help='number of test points (default: %(default)s)')
    parser.add_argument('--b_function_index', type=int, default=0, help='index of benchmark function (default: %(default)s)')
    parser.add_argument('--std', type=float, default=1, help='std of noise added to the training and test data (default: %(default)s)')

    args = parser.parse_args()

    parameter_names = ['F0', 'F0_param', 'f0', 'f0_param', 'kappa_f', 'kappa_g', 'scale_f', 'scale_g', 'sigma', 'alpha', 'beta']
    benchmark_function_names = ['flat', 'skew', 'parabola', 'abs', 'sine', 'step']
    benchmark_function = benchmark_function_names[args.b_function_index]


    # Paths to data and model
    path = '6_Experiments/63_Ushaped_benchmark_functions/HS_2deriv_model/'
    resultpath = path + 'results/' + benchmark_function +'/'
    samplepath = resultpath + 'experiment_samples/'

    os.makedirs(resultpath, exist_ok = True)
    os.makedirs(samplepath, exist_ok = True)

    # Load code
    with open(path + "HSGP_add_GP_second_deriv.stan", "r") as f:
        simulation_code = f.read()

    m_list = [5, 10, 20]
    L_list = [10, 20, 30]

    best_m = 3
    best_L = 10
    best_RMSE = 1.275

    with open(resultpath+'model_selection_results.txt', 'w') as file:
        file.write(f"model: {benchmark_function}\n")
    
    for m in m_list:
        for L in L_list:
            print(f'm = {m} and L = {L}')
            RMSE_mean, RMSE_std, lpd_mean, lpd_std, rhat_test, RMSE_array, lpd_array = run_simulation(m, L, simulation_code, parameter_names)
            print(f"Experiments done for m = {m} and L = {L}.")
      
            with open(resultpath+'model_selection_results.txt', 'a') as file:
                file.write(f"m: {m}, L: {L}, RMSE: {RMSE_mean:.3f}, RMSE_std: {RMSE_std:.3f}, lpd: {lpd_mean:.3f}, lpd_std: {lpd_std}, convergence test: {rhat_test} \n")

            np.save(samplepath + f'RMSE_m_{m}_L_{L}.npy', RMSE_array)
            np.save(samplepath + f'lpd_m_{m}_L_{L}.npy', lpd_array)

            if RMSE_mean < best_RMSE:
                best_RMSE = RMSE_mean
                best_m = m
                best_L = L
            

    # Save the best results
    best_configuration = {'m': best_m, 'L': best_L} 
    with open(resultpath+'model_selection_results.pkl', 'wb') as file:
        pickle.dump(best_configuration, file)

