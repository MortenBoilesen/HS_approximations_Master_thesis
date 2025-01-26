import stan
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils import compute_RMSE, compute_lpd, benchmark_functions, compute_rhat


def run_simulation(m, L, simulation_code, args, parameter_names):
    print(f'm = {m} and L = {L}')
    rhat_test = np.zeros(args.num_experiments, dtype=bool)
    lpd_array = np.zeros(args.num_experiments)
    RMSE_array = np.zeros(args.num_experiments)

    for i in range(args.num_experiments):
        np.random.seed(i)
        # Generate data
        x_train = np.random.uniform(0, 10, args.num_train)
        x_test = np.random.uniform(0, 10, args.num_test)
        y_train = benchmark_functions(args.b_function_index, x_train) + np.random.normal(0, args.std, args.num_train)
        y_test = benchmark_functions(args.b_function_index, x_test) + np.random.normal(0, args.std, args.num_test)

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
        posterior = stan.build(simulation_code, data=simulation_data, random_seed=i)

        fit = posterior.sample(num_chains=args.num_chains, num_samples=args.num_samples, num_warmup=args.num_warmup)
        
        param_list = []
        for p in parameter_names:
            param = fit[p]
            for j in range(len(param)):
                param_list.append(param[j])

        params = np.stack(param_list)
        # np.save(f'{path}results/{benchmark_function}/params_{i}.npy', params)
        rs = compute_rhat(parameter_array=params, num_chains=args.num_chains, num_samples=args.num_samples)
        
        # np.save(f'{path}results/{benchmark_function}/rhat_{i}.npy', rs)
        rhat_test[i] = np.prod((rs < 1.1), dtype=bool)
        #indsæt rhat check her
        ftest = fit["f_test"]
        sigma = fit["sigma"]
        lpd_array[i] = compute_lpd(y_test, ftest, sigma)
        RMSE_array[i] = compute_RMSE(y_test, ftest)

    # lpd_mean = np.mean(lpd_array[rhat_test])
    # RMSE_mean = np.mean(RMSE_array[rhat_test])
    # lpd_std = np.std(lpd_array[rhat_test])
    # RMSE_std = np.std(RMSE_array[rhat_test])
    lpd_mean = np.mean(lpd_array)
    RMSE_mean = np.mean(RMSE_array)
    lpd_std = np.std(lpd_array)
    RMSE_std = np.std(RMSE_array)

    return m, L, RMSE_mean, RMSE_std, lpd_mean, lpd_std, rhat_test


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

    # Paths to data and model
    path = '6_Experiments/61_Monotonic_benchmark_functions/HS_model/'

    parameter_names = ['f0', 'f0_param', 'kappa', 'scale', 'sigma']
    benchmark_function_names = ['flat', 'sinusoidal', 'step', 'linear', 'exp', 'logical']
    benchmark_function = benchmark_function_names[args.b_function_index]
    # Load code
    with open(path + "HSGP_monotonic.stan", "r") as f:
        simulation_code = f.read()

    m_array = [2, 3]#, 5, 10]#, 15, 20, 25, 30]
    L_array = [10, 15]#, 20, 30]
    results = []
    for m in m_array:
        for L in L_array:
            result = run_simulation(m, L, simulation_code, args, parameter_names)
            results.append(result)

    best_configuration_results = np.array([100, 100, 100, 100, 0, 0], dtype=float)

    with open(f'6_Experiments/61_Monotonic_benchmark_functions/HS_model/results/{benchmark_function}/results.txt', 'w') as file:
        file.write(f"model: {benchmark_function}\n")

        for result in results:
            m, L, RMSE_mean, RMSE_std, lpd_mean, lpd_std, rhat_test = result
            file.write(f"m: {m}, L: {L}, RMSE: {RMSE_mean:.3f}, RMSE_std: {RMSE_std:.3f}, lpd: {lpd_mean:.3f}, lpd_std: {lpd_std}, convergence test: {rhat_test} \n")

            if RMSE_mean < best_configuration_results[0]:
                best_configuration_results[0] = RMSE_mean
                best_configuration_results[1] = RMSE_std
                best_configuration_results[2] = lpd_mean
                best_configuration_results[3] = lpd_std
                best_configuration_results[4] = m
                best_configuration_results[5] = L

        # Save the best results
        np.save(f'{path}results/{benchmark_function}/results.npy', best_configuration_results)

        file.write(f"\n Optimization done with m = {best_configuration_results[4]}, L = {best_configuration_results[5]} and RMSE = {best_configuration_results[0]:.3f}+-{best_configuration_results[1]:.3f} and LPD {best_configuration_results[2]:.3f}+-{best_configuration_results[3]:.3f} as the best configuration \n")

        
        print(f"Optimization done with m = {best_configuration_results[4]}, L = {best_configuration_results[5]} and RMSE = {best_configuration_results[0]:.3f}+-{best_configuration_results[1]:.3f} and LPD {best_configuration_results[2]:.3f}+-{best_configuration_results[3]:.3f} as the best configuration")

    
    