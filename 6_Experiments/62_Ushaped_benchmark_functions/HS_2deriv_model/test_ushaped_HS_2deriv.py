import stan
import numpy as np
import sys
import pickle
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils import compute_RMSE, compute_lpd, benchmark_ushaped, compute_rhat


def run_simulation(m, L, parameter_names, seed_0 = 0):
    print(f'm = {m} and L = {L}')

    # Load code
    with open(path + "HSGP_add_GP_second_deriv.stan", "r") as f:
        simulation_code = f.read()


    rhat_test = np.zeros(args.num_experiments, dtype=bool)
    lpd_array = np.zeros(args.num_experiments)
    RMSE_array = np.zeros(args.num_experiments)
    rs_array = np.zeros((args.num_experiments, args.num_chains, len(parameter_names) -2 + 2*m ))
    params_array = np.zeros((args.num_experiments, len(parameter_names) -2 + 2*m, args.num_chains*args.num_samples))

    for i in range(args.num_experiments):
        np.random.seed(seed_0 + i+19)
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

        if i == args.num_experiments - 1:
            with open(path + "HSGP_add_GP_second_deriv_pred.stan", "r") as f:
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
    gp_part = fit['GP_pred']
    f0_pred = fit['f0_pred']
    F0_pred = fit['F0']
    
    return RMSE_mean, RMSE_std, lpd_mean, lpd_std, rhat_test, x_train, y_train, fpred, sigma, params_array, rs_array, gp_part, f0_pred, F0_pred


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_chains', type=int, default=3, help='number of chains in HMC sampler (default: %(default)s)')
    parser.add_argument('--num_samples', type=int, default=3000, help='number of samples in each chain (default: %(default)s)')
    parser.add_argument('--num_warmup', type=int, default=1000, help='number of samples in warm up (default: %(default)s)')
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
    params_rhat_path = resultpath + 'params_and_rhat/'



    # Load best parameters
    with open(resultpath+'model_selection_results.pkl', 'rb') as file:
        best_configuration = pickle.load(file)
    
    # resultpath = path + 'results_big_f0_prior_exp/' + benchmark_function +'/'
    params_rhat_path = resultpath + 'params_and_rhat/'

    os.makedirs(resultpath, exist_ok = True)
    os.makedirs(params_rhat_path, exist_ok = True)

    best_configuration_results = np.zeros(4)
    
    RMSE_mean, RMSE_std, lpd_mean, lpd_std, rhat_test, x_train, y_train, fpred, sigma, params_array, rs_array, gp_part, f0_pred, F0_pred = run_simulation(best_configuration['m'], best_configuration['L'], parameter_names, seed_0 = 1234*(args.b_function_index + 1))

    best_configuration_results[0] = RMSE_mean
    best_configuration_results[1] = RMSE_std 
    best_configuration_results[2] = lpd_mean 
    best_configuration_results[3] = lpd_std 

   # Save the best results
    np.save(resultpath+f'{args.num_samples}_test_results.npy', (best_configuration_results))
    np.save(resultpath+f'{args.num_samples}_test_rhat.npy', (rhat_test))
    np.save(resultpath+f'{args.num_samples}_test_xtrain.npy', (x_train))
    np.save(resultpath+f'{args.num_samples}_test_ytrain.npy', (y_train))
    np.save(resultpath+f'{args.num_samples}_test_fpred.npy', (fpred))
    np.save(resultpath+f'{args.num_samples}_test_sigma.npy', (sigma))
    np.save(resultpath+f'{args.num_samples}_test_gppart.npy', (gp_part))
    np.save(resultpath+f'{args.num_samples}_test_bigf0.npy', (F0_pred))
    np.save(resultpath+f'{args.num_samples}_test_smallf0.npy', (f0_pred))

    np.save(params_rhat_path+f'{args.num_samples}_test_params_array.npy', (params_array))
    np.save(params_rhat_path+f'{args.num_samples}_test_rhat_array.npy', (rs_array))
