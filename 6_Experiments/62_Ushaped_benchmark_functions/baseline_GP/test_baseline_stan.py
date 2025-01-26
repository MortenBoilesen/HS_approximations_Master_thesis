import stan
import numpy as np
import sys
import pickle
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils import compute_RMSE, compute_lpd, benchmark_ushaped, compute_rhat


def run_simulation(parameter_names, seed_0 = 0):

    # Load code
    with open(path + "gp.stan", "r") as f:
        simulation_code = f.read()


    rhat_test = np.zeros(args.num_experiments, dtype=bool)
    lpd_array = np.zeros(args.num_experiments)
    RMSE_array = np.zeros(args.num_experiments)
    rs_array = np.zeros((args.num_experiments, args.num_chains, len(parameter_names)  ))
    params_array = np.zeros((args.num_experiments, len(parameter_names), args.num_chains*args.num_samples))

    for i in range(args.num_experiments):
        np.random.seed(seed_0 + i)
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
            "ytrain": y_train
        }

        if i == args.num_experiments - 1:
            with open(path + "gp_pred.stan", "r") as f:
                simulation_code = f.read()
            
            num_pred = 100
            x_pred = np.linspace(-5,5, num_pred)

            simulation_data["num_pred"] = num_pred
            simulation_data["xpred"] = x_pred

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
        
        # np.save(f'{path}results/{benchmark_function}/rhat_{i}.npy', rs)
        rhat_test[i] = np.prod((rs < 1.1), dtype=bool)
        #indsæt rhat check her
        ftest = fit["f_test"]
        sigma = fit["sigma"]
        lpd_array[i] = compute_lpd(y_test, ftest, sigma)
        RMSE_array[i] = compute_RMSE(y_test, ftest)
        rs_array[i] = rs
        params_array[i] = params

    lpd_mean = np.mean(lpd_array)
    RMSE_mean = np.mean(RMSE_array)
    lpd_std = np.std(lpd_array)
    RMSE_std = np.std(RMSE_array)

    fpred = fit["f_pred"]
    return RMSE_mean, RMSE_std, lpd_mean, lpd_std, rhat_test, x_train, y_train, fpred, sigma, params_array, rs_array


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_chains', type=int, default=4, help='number of chains in HMC sampler (default: %(default)s)')
    parser.add_argument('--num_samples', type=int, default=10000, help='number of samples in each chain (default: %(default)s)')
    parser.add_argument('--num_warmup', type=int, default=10000, help='number of samples in warm up (default: %(default)s)')
    parser.add_argument('--num_experiments', type=int, default=20, help='number of experiments (default: %(default)s)')
    parser.add_argument('--num_train', type=int, default=15, help='number of training points (default: %(default)s)')
    parser.add_argument('--num_test', type=int, default=100, help='number of test points (default: %(default)s)')
    parser.add_argument('--b_function_index', type=int, default=0, help='index of benchmark function (default: %(default)s)')
    parser.add_argument('--std', type=float, default=1, help='std of noise added to the training and test data (default: %(default)s)')

    args = parser.parse_args()

    parameter_names = ['kappa', 'scale', 'sigma']
    benchmark_function_names = ['flat', 'skew', 'parabola', 'abs', 'sine', 'step']
    benchmark_function = benchmark_function_names[args.b_function_index]


    # Paths to data and model
    path = '6_Experiments/63_Ushaped_benchmark_functions/baseline_GP/'
    resultpath = path + 'results_stan/' + benchmark_function +'/'

    os.makedirs(resultpath, exist_ok = True)



    # resultpath = path + 'results_less_noise/' + benchmark_function +'/'
    params_rhat_path = resultpath + 'params_and_rhat/'

    # os.makedirs(resultpath, exist_ok = True)
    os.makedirs(params_rhat_path, exist_ok = True)

    best_configuration_results = np.zeros(4)
    
    RMSE_mean, RMSE_std, lpd_mean, lpd_std, rhat_test, x_train, y_train, fpred, sigma, params_array, rs_array = run_simulation(parameter_names, seed_0 = 12345*(args.b_function_index+1))

    best_configuration_results[0] = RMSE_mean
    best_configuration_results[1] = RMSE_std 
    best_configuration_results[2] = lpd_mean 
    best_configuration_results[3] = lpd_std 

   # Save the best results
    np.save(resultpath+f'test_results.npy', (best_configuration_results))
    np.save(resultpath+f'test_rhat.npy', (rhat_test))
    np.save(resultpath+f'test_xtrain.npy', (x_train))
    np.save(resultpath+f'test_ytrain.npy', (y_train))
    np.save(resultpath+f'test_fpred.npy', (fpred))
    np.save(resultpath+f'test_sigma.npy', (sigma))

    np.save(params_rhat_path+f'test_params_array.npy', (params_array))
    np.save(params_rhat_path+f'test_rhat_array.npy', (rs_array))
