import stan
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pickle
from utils import compute_RMSE, compute_lpd, compute_rhat


def run_simulation(num_virtual, simulation_code, fraction, num_total, x_train, y_train, parameter_names, seed_0 = 0 ):
    rhat_test = np.zeros(args.num_folds, dtype=bool)
    lpd_array = np.zeros(args.num_folds)
    RMSE_array = np.zeros(args.num_folds)

    shuffle_idx = np.arange(num_total)
    np.random.shuffle(shuffle_idx)

    xvirtual = np.linspace(-1.8, 1.8, num_virtual)
    yvirtual = np.zeros(num_virtual, dtype=int) # Zeros because we want the derirative to be negative


    for i in range(args.num_folds):
        num_val = num_total // args.num_folds

        val_idx = shuffle_idx[i*num_val:(i+1)*num_val]
        rest_idx = np.setdiff1d(shuffle_idx, val_idx)
        num_train = int(len(rest_idx) * fraction / 8)
        np.random.shuffle(rest_idx)
        train_idx = rest_idx[:num_train]
     
        y_val = y_train[val_idx]

        simulation_data = {
            "num_test": len(x_train[val_idx]),
            "xtest": x_train[val_idx],
            "ytest": y_val,

            "num_train": len(x_train[train_idx]),
            "xtrain": x_train[train_idx],
            "ytrain": y_train[train_idx],

            "num_virtual": num_virtual,
            "xvirtual": xvirtual,
            "yvirtual": yvirtual,
            "nu": args.nu,

            "jitter": 1e-8,
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

        f0 = fit["f0"]
        fval = f0 + fit["f_test"]
        sigma = fit["sigma"]
        lpd_array[i] = compute_lpd(y_val, fval, sigma)
        RMSE_array[i] = compute_RMSE(y_val, fval)

    lpd_mean = np.mean(lpd_array)
    RMSE_mean = np.mean(RMSE_array)
    lpd_std = np.std(lpd_array)
    RMSE_std = np.std(RMSE_array)

    return RMSE_mean, RMSE_std, lpd_mean, lpd_std, rhat_test, RMSE_array, lpd_array


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--R', type=int, default=2, help='What run (default: %(default)s)')
    parser.add_argument('--num_chains', type=int, default=3, help='number of chains in HMC sampler (default: %(default)s)')
    parser.add_argument('--num_samples', type=int, default=500, help='number of samples in each chain (default: %(default)s)')
    parser.add_argument('--num_warmup', type=int, default=500, help='number of samples in warm up (default: %(default)s)')
    parser.add_argument('--num_folds', type=int, default=5, help='number of folds in k-fold validation (default: %(default)s)')
    parser.add_argument('--num_train', type=int, default=15, help='number of training points (default: %(default)s)')
    parser.add_argument('--num_test', type=int, default=100, help='number of test points (default: %(default)s)')
    parser.add_argument('--fraction', type=float, default=1.0, help='fraction of data to use measured in eigths (default: %(default)s)')
    parser.add_argument('--nu', type=float, default=1e2, help='controls the strictness of monotonicity information in the virtual points (default: %(default)s)')

    args = parser.parse_args()

    parameter_names = ['scale', 'kappa', 'sigma', 'eta', 'f0']

    # Paths to data and model
    path = f'6_Experiments/64_Ushaped_data/Runs/Run_{args.R}/'
    datapath = path + '/00data/'
    resultpath = path + f'VP_model/{args.fraction}/'
    samplepath = resultpath + 'experiment_samples/'

    os.makedirs(resultpath, exist_ok = True)
    os.makedirs(samplepath, exist_ok = True)

    # Load code
    with open("6_Experiments/64_Ushaped_data/VP_model/gp_convex_gaussian_virtual.stan", "r") as f:
        simulation_code = f.read()

    # Load training data
    x_train = np.load(datapath+f'x_train.npy')
    num_total = len(x_train)
    y_train = np.load(datapath+f'y_train.npy')

    nvp_list = [5, 10, 20, 50]
    best_num_virtual = 100
    best_RMSE = 1000

    with open(resultpath+'model_selection_results.txt', 'w') as file:
        file.write(f"data fraction: {args.fraction}\n")
    
    for num_virtual in nvp_list:
        print(f'number of virtual points: {num_virtual}')
        result = run_simulation(num_virtual, simulation_code, args.fraction, num_total, x_train, y_train, parameter_names)
        print(f'Experiment done for number of virtual points: {num_virtual}')

        RMSE_mean, RMSE_std, lpd_mean, lpd_std, rhat_test, RMSE_array, lpd_array = result
        with open(resultpath+'model_selection_results.txt', 'a') as file:
            file.write(f"num virtual: {num_virtual}, RMSE: {RMSE_mean:.3f}, RMSE_std: {RMSE_std:.3f}, lpd: {lpd_mean:.3f}, lpd_std: {lpd_std}, convergence test: {rhat_test} \n")

        np.save(samplepath + f'RMSE_num_v_{num_virtual}.npy', RMSE_array)
        np.save(samplepath + f'lpd_num_v_{num_virtual}.npy', lpd_array)

        if RMSE_mean < best_RMSE:
            best_RMSE = RMSE_mean
            best_num_virtual = num_virtual

    # Save the best results
    best_configuration = {'num_virtual': best_num_virtual} 
    with open(resultpath+'model_selection_results.pkl', 'wb') as file:
        pickle.dump(best_configuration, file)
