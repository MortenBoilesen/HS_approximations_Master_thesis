import stan
import numpy as np
import time
from joblib import Parallel, delayed
import psutil
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils import compute_RMSE, compute_lpd, benchmark_functions, compute_rhat

def run_simulation(num_virtual, args, simulation_code, parameter_names):
    print(f'number of virtual points: {num_virtual}')
    
    start_time = time.time()

    rhat_test = np.zeros(args.num_experiments, dtype=bool)
    lpd_array = np.zeros(args.num_experiments)
    RMSE_array = np.zeros(args.num_experiments)

    xvirtual = np.linspace(0, 10, num_virtual)
    yvirtual = np.ones(num_virtual, dtype=int)

    
    for i in range(args.num_experiments):
        # Generate data
        x_train = np.random.uniform(0, 10, args.num_train)
        x_test = np.random.uniform(0, 10, args.num_test)
        y_train = benchmark_functions(args.b_function_index, x_train) + np.random.normal(0, args.std, args.num_train)
        y_test = benchmark_functions(args.b_function_index, x_test) + np.random.normal(0, args.std, args.num_test)
     
        np.random.seed(i)
        simulation_data = {                  
                    "num_train": args.num_train,
                    "xtrain": x_train,
                    "ytrain": y_train,

                    "num_test": args.num_test,
                    "xtest": x_test,
                    "ytest": y_test,

                    "num_virtual": num_virtual,
                    "xvirtual": xvirtual,
                    "yvirtual": yvirtual,

                    "nu": args.nu,
                    "jitter": 1e-8,
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
        rhat_test[i] = np.prod((rs < 1.1), dtype=bool)
        ftest = fit["f_test"]
        sigma = fit["sigma"]
        lpd_array[i] = compute_lpd(y_test, ftest, sigma)
        RMSE_array[i] = compute_RMSE(y_test, ftest)

    
    lpd_mean = np.mean(lpd_array)
    RMSE_mean = np.mean(RMSE_array)
    lpd_std = np.std(lpd_array)
    RMSE_std = np.std(RMSE_array)  
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Experiments done in {int(elapsed_time // 60)} minutes and {int(elapsed_time % 60)} seconds for num_virtual = {num_virtual} and benchmark function = {args.b_function_index}..")


    return num_virtual, RMSE_mean, RMSE_std, lpd_mean, lpd_std, rhat_test

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
    parser.add_argument('--nu', type=float, default=1e2, help='controls the strictness of monotonicity information in the virtual points (default: %(default)s)')

    args = parser.parse_args()

    ## Paths to data and model
    path = '6_Experiments/61_Monotonic_benchmark_functions/VP_model/'
    datapath = '6_Experiments/61_Monotonic_benchmark_functions/00data/'

    parameter_names = ['scale', 'kappa', 'sigma', 'eta', 'f0']
    benchmark_function_names = ['flat', 'sinusoidal', 'step', 'linear', 'exp', 'logical']
    benchmark_function = benchmark_function_names[args.b_function_index]
    # Load code
    with open(path+"gp_mono_gaussian_virtual.stan", "r") as f:
        simulation_code = f.read()

    current_process = psutil.Process()
    subproc_before = set([process_n.pid for process_n in current_process.children(recursive=True)])

    with open(f'6_Experiments/61_Monotonic_benchmark_functions/VP_model/results/{benchmark_function}/results.txt', 'w') as file:
        file.write(f"model: {benchmark_function}\n")

        results = Parallel(n_jobs=-1)(delayed(run_simulation)(num_virtual, args, simulation_code, parameter_names) 
                                      for num_virtual in [10, 20, 50, 100] 
                                      )

        best_configuration_results = np.array([100, 100, 100, 100, 0], dtype=float)

        for result in results:
            num_virtual, RMSE_mean, RMSE_std, lpd_mean, lpd_std, rhat_test = result
            file.write(f"num_virtual: {num_virtual}, RMSE: {RMSE_mean:.3f}, RMSE_std: {RMSE_std:.3f}, lpd: {lpd_mean:.3f}, lpd_std: {lpd_std}, convergence test: {rhat_test} \n")

            if RMSE_mean < best_configuration_results[0]:
                best_configuration_results[0] = RMSE_mean
                best_configuration_results[1] = RMSE_std
                best_configuration_results[2] = lpd_mean
                best_configuration_results[3] = lpd_std
                best_configuration_results[4] = num_virtual

        np.save(f'{path}results/{benchmark_function}/results.npy', best_configuration_results)


        file.write(f"\n Optimization done with num_virtual = {best_configuration_results[4]} and RMSE = {best_configuration_results[0]:.3f}+-{best_configuration_results[1]:.3f} and LPD {best_configuration_results[2]:.3f}+-{best_configuration_results[3]:.3f} as the best configuration \n")
    print(f"Optimization done with num_virtual = {best_configuration_results[4]} and RMSE = {best_configuration_results[0]:.3f}+-{best_configuration_results[1]:.3f} and LPD {best_configuration_results[2]:.3f}+-{best_configuration_results[3]:.3f} as the best configuration")

    subproc_after = set([process_n.pid for process_n in current_process.children(recursive=True)])
    for subproc in subproc_after - subproc_before:
        psutil.Process(subproc).terminate()