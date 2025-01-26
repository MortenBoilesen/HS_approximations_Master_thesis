import stan
import numpy as np
import sys
import pickle
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils import compute_RMSE, compute_lpd, benchmark_ushaped, compute_rhat

def run_simulation(m, L, parameter_names, index, seed_0 = 0):
    print(f'm = {m} and L = {L}')

    # Load code
    with open(path + "HSGP_add_GP_second_deriv.stan", "r") as f:
        simulation_code = f.read()

    rs_array = np.zeros((args.num_chains, len(parameter_names) -2 + 2*m ))
    params_array = np.zeros((len(parameter_names) -2 + 2*m, args.num_chains*args.num_samples))

    np.random.seed(seed_0 + index)
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

    if index == args.num_experiments - 1:
        with open(path + "HSGP_add_GP_second_deriv_pred.stan", "r") as f:
            simulation_code = f.read()
        
        num_pred = 100
        x_pred = np.linspace(-5,5, num_pred)

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
    
    # np.save(f'{path}results/{benchmark_function}/rhat_{i}.npy', rs)
    rhat_test = np.prod((rs < 1.1), dtype=bool)
    #indsæt rhat check her
    ftest = fit["f_test"]
    sigma = fit["sigma"]
    lpd = compute_lpd(y_test, ftest, sigma)
    RMSE = compute_RMSE(y_test, ftest)
    rs_array = rs
    params_array = params


    fpred = 0
    gp_part = 0
    f0_pred = 0
    F0_pred = 0

    if index == args.num_experiments - 1:

        fpred = fit["f_pred"]
        gp_part = fit['GP_pred']
        f0_pred = fit['f0_pred']
        F0_pred = fit['F0']
    
    
    return RMSE, lpd, rhat_test, x_train, y_train, sigma, params_array, rs_array, fpred, gp_part, f0_pred, F0_pred


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_chains', type=int, default=4, help='number of chains in HMC sampler (default: %(default)s)')
    parser.add_argument('--num_samples', type=int, default=500, help='number of samples in each chain (default: %(default)s)')
    parser.add_argument('--num_warmup', type=int, default=500, help='number of samples in warm up (default: %(default)s)')
    parser.add_argument('--num_experiments', type=int, default=20, help='number of experiments (default: %(default)s)')
    parser.add_argument('--num_train', type=int, default=15, help='number of training points (default: %(default)s)')
    parser.add_argument('--num_test', type=int, default=100, help='number of test points (default: %(default)s)')
    parser.add_argument('--b_function_index', type=int, default=0, help='index of benchmark function (default: %(default)s)')
    parser.add_argument('--std', type=float, default=1, help='std of noise added to the training and test data (default: %(default)s)')
    parser.add_argument('--index', type=int, default=0, help='index of experiment (default: %(default)s)')

    args = parser.parse_args()

    parameter_names = ['F0', 'F0_param', 'f0', 'f0_param', 'kappa_f', 'kappa_g', 'scale_f', 'scale_g', 'sigma', 'alpha', 'beta']
    benchmark_function_names = ['flat', 'skew', 'parabola', 'abs', 'sine', 'step']
    benchmark_function = benchmark_function_names[args.b_function_index]


    # Paths to data and model
    path = '6_Experiments/63_Ushaped_benchmark_functions/HS_2deriv_model/'
    resultpath = path + 'results/' + benchmark_function +'/'
    params_rhat_path = resultpath + 'params_and_rhat/'

    os.makedirs(resultpath, exist_ok = True)
    os.makedirs(params_rhat_path, exist_ok = True)

    # Load best parameters
    with open(resultpath+'model_selection_results.pkl', 'rb') as file:
        best_configuration = pickle.load(file)

    resultpath = path + 'results/' + benchmark_function +f'/{args.index}/'
    params_rhat_path = resultpath + 'params_and_rhat/'

    os.makedirs(resultpath, exist_ok = True)
    os.makedirs(params_rhat_path, exist_ok = True)

    RMSE, lpd, rhat_test, x_train, y_train, sigma, params_array, rs_array, fpred, gp_part, f0_pred, F0_pred = run_simulation(best_configuration['m'], best_configuration['L'], parameter_names, args.index, seed_0 = 1234*(args.b_function_index + 1))

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

        # Save the best results
        best_configuration_results = np.zeros(4)
        best_configuration_results[0] = RMSE_mean
        best_configuration_results[1] = RMSE_std 
        best_configuration_results[2] = lpd_mean 
        best_configuration_results[3] = lpd_std 

        # Save the best results
        np.save(resultpath+'10000_test_results.npy', (best_configuration_results))
        np.save(resultpath+'10000_test_rhat.npy', (rhat_test))
        np.save(resultpath+'10000_test_xtrain.npy', (x_train))
        np.save(resultpath+'10000_test_ytrain.npy', (y_train))
        np.save(resultpath+'10000_test_fpred.npy', (fpred))
        np.save(resultpath+'10000_test_gppart.npy', (gp_part))
        np.save(resultpath+'10000_test_f0_pred.npy', (f0_pred))
        np.save(resultpath+'10000_test_bigf0_pred.npy', (F0_pred))
        np.save(resultpath+'10000_test_sigma.npy', (sigma))
        np.save(params_rhat_path+'10000_test_params.npy', (params_array))
        np.save(params_rhat_path+'10000_test_rhat.npy', (rs_array))
    else:
        np.save(resultpath+f'10000_test_RMSE.npy', (RMSE))
        np.save(resultpath+f'10000_test_lpd.npy', (lpd)) 
        np.save(resultpath+f'10000_test_rhat.npy', (rhat_test))