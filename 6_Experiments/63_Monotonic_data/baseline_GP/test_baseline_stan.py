import stan
import numpy as np
import sys
import os
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils import compute_RMSE, compute_lpd, compute_rhat


def run_simulation(simulation_code, parameter_names, x_train, x_test, x_pred, y_train, y_test):
    num_train = len(x_train)
    num_test = len(x_test)
    num_pred = len(x_pred)

    np.random.seed(1)
    simulation_data = {
                "num_train": num_train,
                "xtrain": x_train,
                "ytrain": y_train,

                "num_test": num_test,
                "xtest": x_test,
                "ytest": y_test,

                "num_pred": num_pred,
                "xpred": x_pred,
                }

# Build model
    posterior = stan.build(simulation_code, data=simulation_data, random_seed= 1)
    fit = posterior.sample(num_chains=args.num_chains, num_samples=args.num_samples, num_warmup=args.num_warmup)
    print('fit done')
    param_list = []
    for p in parameter_names:
        param = fit[p]
        for j in range(len(param)):
            param_list.append(param[j])
    print('parameters done')
    params = np.stack(param_list)

    rs = compute_rhat(parameter_array=params, num_chains=args.num_chains, num_samples=args.num_samples)
    rhat_test = np.prod((rs < 1.1), dtype=bool)
    f_test = fit["f_test"]
    sigma = fit["sigma"]
    f_pred = fit['f_pred']
    lpd_array = compute_lpd(y_test, f_test, sigma)
    RMSE_array = compute_RMSE(y_test, f_test)
    print('rs and lpd and rmse done')

    return RMSE_array, lpd_array, rhat_test, f_pred, sigma, params, rs

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_chains', type=int, default=4, help='number of chains in HMC sampler (default: %(default)s)')
    parser.add_argument('--num_samples', type=int, default=500, help='number of samples in each chain (default: %(default)s)')
    parser.add_argument('--num_warmup', type=int, default=500, help='number of samples in warm up (default: %(default)s)')

    args = parser.parse_args()

    parameter_names = ['scale', 'kappa', 'sigma']

    path = '5_Shape_constrained_modelling/53_Relaxing_convexity/baseline_GP/'
    datapath = '5_Shape_constrained_modelling/53_Relaxing_convexity/00data/'
    resultpath = path + 'results/stan/'
    params_rhat_path = resultpath + 'params_and_rhat/'

    os.makedirs(resultpath, exist_ok = True)
    os.makedirs(params_rhat_path, exist_ok = True)

    x_train = np.load(datapath + 'Fertility_rate/x_train.npy')
    y_train = np.load(datapath + 'Fertility_rate/y_train.npy')
    x_test = np.load(datapath + 'Fertility_rate/x_test.npy')
    y_test = np.load(datapath + 'Fertility_rate/y_test.npy')
    x_pred = np.linspace(np.min(x_train), np.max(x_test), 100)

    # Load code
    with open(path+"gp_pred.stan", "r") as f:
        simulation_code = f.read()

    print('Running HMC')
    RMSE_array, lpd_array, rhat_test, f_pred, sigma, params, rs = run_simulation(simulation_code, parameter_names, x_train, x_test, x_pred, y_train, y_test)
    print('HMC Finished.')
    print('Saving parameters to ' + resultpath)
    best_configuration_results = np.zeros(2)
    best_configuration_results[0] = RMSE_array
    best_configuration_results[1] = lpd_array

    np.save(resultpath+f'{args.num_samples}_test_results.npy', (best_configuration_results))
    np.save(resultpath+f'{args.num_samples}_test_fpred.npy', f_pred)
    np.save(resultpath+f'{args.num_samples}_test_sigma.npy', sigma)
    np.save(params_rhat_path+f'{args.num_samples}_test_rhat.npy', (rs))
    np.save(params_rhat_path+f'{args.num_samples}_test_params.npy', (params))

    print('Parameters saved.')