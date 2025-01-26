import stan
import numpy as np
import sys
import os
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils import compute_RMSE, compute_lpd, benchmark_functions, compute_rhat

def run_simulation(num_virtual, parameter_names, index, seed_0 = 0):
    print(f'number of virtual points: {num_virtual}')

    # Load code
    with open(path+"gp_mono_gaussian_virtual.stan", "r") as f:
        simulation_code = f.read()

    
    rs_array = np.zeros((args.num_chains, len(parameter_names) - 1 + args.num_train+num_virtual))
    params_array = np.zeros((len(parameter_names) - 1 + args.num_train+num_virtual, args.num_chains*args.num_samples))

    xvirtual = np.linspace(0, 10, num_virtual)
    yvirtual = np.ones(num_virtual, dtype=int)
    
    np.random.seed(index + seed_0)   
    # Generate data
    x_train = np.random.uniform(0, 10, args.num_train)
    x_test = np.random.uniform(0, 10, args.num_test)
    y_train = benchmark_functions(args.b_function_index, x_train) + np.random.normal(0, args.std, args.num_train)
    y_test = benchmark_functions(args.b_function_index, x_test) + np.random.normal(0, args.std, args.num_test)
    

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

    if index == args.num_experiments -1:
        with open(path + "gp_mono_gaussian_virtual_pred.stan", "r") as f:
            simulation_code = f.read()

            num_pred = 100
            x_pred = np.linspace(0, 10, num_pred)

            simulation_data["num_pred"] = num_pred
            simulation_data["xpred"] = x_pred


# Build model
    posterior = stan.build(simulation_code, data=simulation_data, random_seed=index + seed_0)

    fit = posterior.sample(num_chains=args.num_chains, num_samples=args.num_samples, num_warmup=args.num_warmup)

    param_list = []
    for p in parameter_names:
        param = fit[p]
        for j in range(len(param)):
            param_list.append(param[j])

    params = np.stack(param_list)

    rs = compute_rhat(parameter_array=params, num_chains=args.num_chains, num_samples=args.num_samples)
    rhat_test = np.prod((rs < 1.1), dtype=bool)
    f0 = fit['f0']
    ftest = fit["f_test"] + f0
    sigma = fit["sigma"]
    lpd = compute_lpd(y_test, ftest, sigma)
    RMSE = compute_RMSE(y_test, ftest)
    rs_array = rs
    params_array = params
    if index == args.num_experiments -1:
        fpred = fit["f_pred"] + f0
    else:
        fpred = 0
    
    return RMSE, lpd, rhat_test, x_train, y_train, fpred, sigma, rs_array, params_array


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
    parser.add_argument('--nu', type=float, default=1e2, help='controls the strictness of monotonicity information in the virtual points (default: %(default)s)')    
    parser.add_argument('--index', type=int, default=0, help='index of experiment (default: %(default)s)')

    args = parser.parse_args()

    parameter_names = ['scale', 'kappa', 'sigma', 'eta', 'f0']
    benchmark_function_names = ['flat', 'sinusoidal', 'step', 'linear', 'exp', 'logical']
    benchmark_function = benchmark_function_names[args.b_function_index]

    path = '6_Experiments/61_Monotonic_benchmark_functions/VP_model/'

    with open(path + 'results/' + benchmark_function +'/model_selection_results.pkl', 'rb') as file:
        best_configuration = pickle.load(file)

    resultpath = path + 'results/' + benchmark_function +f'/{args.index}/'
    params_rhat_path = resultpath + 'params_and_rhat/'

    os.makedirs(resultpath, exist_ok = True)
    # os.makedirs(params_rhat_path, exist_ok = True)

    # Load optimal model parameters

    RMSE, lpd, rhat_test, x_train, y_train, fpred, sigma, rs_array, params_array = run_simulation(best_configuration['num_virtual'], parameter_names, args.index, seed_0 = 1234*(args.b_function_index + 1))

    if args.index == args.num_experiments -1:
        lpd_array = np.zeros(args.num_experiments)
        RMSE_array = np.zeros(args.num_experiments)

        for experiment in range(args.num_experiments -1):
            resultpath = path + 'results/' + benchmark_function +f'/{experiment}/'
            rmse_sample = np.load(resultpath+f'{args.num_samples}_test_RMSE.npy')
            lpd_sample = np.load(resultpath+f'{args.num_samples}_test_lpd.npy')
            RMSE_array[experiment] = rmse_sample
            lpd_array[experiment] = lpd_sample
        
        RMSE_array[-1] = RMSE
        lpd_array[-1] = lpd

        lpd_mean = np.mean(lpd_array)
        RMSE_mean = np.mean(RMSE_array)
        lpd_std = np.std(lpd_array)
        RMSE_std = np.std(RMSE_array)

        resultpath = path + 'results/' + benchmark_function +'/'
        params_rhat_path = resultpath + 'params_and_rhat/'

        # Save the best results
        best_configuration_results = np.zeros(4)
        best_configuration_results[0] = RMSE_mean
        best_configuration_results[1] = RMSE_std 
        best_configuration_results[2] = lpd_mean 
        best_configuration_results[3] = lpd_std 

        np.save(resultpath+f'{args.num_samples}_test_results.npy', (best_configuration_results))
        np.save(resultpath+f'{args.num_samples}_test_rhat.npy', (rhat_test))
        np.save(resultpath+f'{args.num_samples}_test_xtrain.npy', (x_train))
        np.save(resultpath+f'{args.num_samples}_test_ytrain.npy', (y_train))
        np.save(resultpath+f'{args.num_samples}_test_fpred.npy', (fpred))
        np.save(resultpath+f'{args.num_samples}_test_sigma.npy', (sigma))
        np.save(params_rhat_path+f'{args.num_samples}_test_rhat.npy', (rs_array))
        np.save(params_rhat_path+f'{args.num_samples}_test_params.npy', (params_array))
    else:
        np.save(resultpath+f'{args.num_samples}_test_RMSE.npy', (RMSE))
        np.save(resultpath+f'{args.num_samples}_test_lpd.npy', (lpd)) 
        np.save(resultpath+f'{args.num_samples}_test_rhat.npy', (rhat_test))

