import stan
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import compute_lpd, compute_RMSE

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='baseline_GP', choices=['baseline_GP', 'concave', 'add_GP', 'add_GP_deriv', 'add_GP_second_deriv'], help='What model to use (default: %(default)s)')
    parser.add_argument('--num_chains', type=int, default=3, help='number of chains in HMC sampler (default: %(default)s)')
    parser.add_argument('--num_samples', type=int, default=1000, help='number of samples in each chain (default: %(default)s)')
    parser.add_argument('--num_warmup', type=int, default=500, help='number of samples in each chain (default: %(default)s)')
    parser.add_argument('--m', type=int, default=40, help='number of basis function. (default: %(default)s)')
    parser.add_argument('--L', type=float, default=10, help='size of domain, [-L,L] (default: %(default)s)')

    args = parser.parse_args()

    ## Paths to data and model
    path = '6_Experiments/63_Monotonic_data/'
    resultpath = path+'results/'+args.model+'/'
    os.makedirs(resultpath, exist_ok=True)

    datapath = path+'00data/'

    if args.model == 'baseline_GP':            
        with open(path+'gp_pred.stan', "r") as f:
            simulation_code = f.read()
    else:
        with open(path+"HSGP_"+args.model+'.stan', "r") as f:
            simulation_code = f.read()


    x_train = np.load(datapath+'xtrain.npy')
    y_train = np.load(datapath+'ytrain.npy')

    num_train = len(x_train)

    x_test = np.load(datapath+'xtest.npy')
    y_test = np.load(datapath+'ytest.npy')

    num_test = len(x_test)

    num_pred = 100
    x_pred = np.linspace(-5,5, num_pred)

    print(f"Number of train: {num_train}, number og test: {num_test}, number of pred: {num_pred}")

    if args.model == 'baseline_GP':
            simulation_data = {
                    "num_test": num_test,
                    "xtest": x_test,
                    "ytest": y_test,
                    
                    "num_train": num_train,
                    "xtrain": x_train,
                    "ytrain": y_train,
                    
                    "num_pred": num_pred,
                    "xpred": x_pred,
                   }
    elif args.model == 'concave':
                sort_idx = np.argsort(x_train)
                index = sort_idx[:(num_train // 4)]
                print(index)
                # Create the subset
                x_train_subset = x_train[index]
                y_train_subset = y_train[index]

                X_b = np.c_[np.ones((x_train_subset.shape[0], 1)), x_train_subset]

                # Calculate the optimal theta using the normal equation
                theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train_subset)
                print(theta_best)
                F0_prior_mean = theta_best[0] + theta_best[1] * (-args.L)

                simulation_data = {
                "num_test": num_test,
                "xtest": x_test,
                "ytest": y_test,
                
                "num_train": num_train,
                "xtrain": x_train,
                "ytrain": y_train,
                
                "num_pred": num_pred,
                "xpred": x_pred,

                "F0_prior_mean": F0_prior_mean,
                "num_basis_functions": args.m,
                "L": args.L,

                "jitter": 1e-8
                }


    else:
        simulation_data = {
                        "num_test": num_test,
                        "xtest": x_test,
                        "ytest": y_test,
                        
                        "num_train": num_train,
                        "xtrain": x_train,
                        "ytrain": y_train,
                        
                        "num_pred": num_pred,
                        "xpred": x_pred,

                        "num_basis_functions": args.m,
                        "L": args.L,

                        "jitter": 1e-8
                        }

    # Build model
    posterior = stan.build(simulation_code, data=simulation_data, random_seed=1)

    fit = posterior.sample(num_chains=args.num_chains, num_samples=args.num_samples, num_warmup=args.num_warmup)

    # Save samples
    ftest = fit["f_test"]
    fpred = fit["f_pred"]
    sigma = fit["sigma"]

    lpd = compute_lpd(y_test=y_test, f_test=ftest, sigma=sigma)
    RMSE = compute_RMSE(y_test, ftest)

    np.save(resultpath+'fpred.npy', fpred)
    np.save(resultpath+'sigma.npy', sigma)
    np.save(resultpath+'results.npy', np.array([RMSE, lpd]))

    params = {}

    # params['f0'] = fit["f0"]
    # params['F0'] = fit["F0"]
    # params['f0_param'] = fit["f0_param"]
    # params['F0_param'] = fit["F0_param"]
    # params['alpha'] = fit['alpha_scaled']
    # params['sigma'] = fit["sigma"]

    # if args.model == 'concave':
    #     params['kappa'] = fit["kappa"]
    #     params['scale'] = fit["scale"]
    # else:
    #     params['beta'] = fit["beta_scaled"]
    #     params['kappa_f'] = fit["kappa_f"]
    #     params['scale_f'] = fit["scale_f"]
    #     params['kappa_g'] = fit["kappa_g"]
    #     params['scale_g'] = fit["scale_g"]

    # import pickle
    # with open(resultpath+'params.pkl', 'wb') as handle:
    #     pickle.dump(params, handle)

    if args.model == 'add_GP':
        gppred = fit["gp_pred"]
        np.save(resultpath+'gppred.npy', gppred)

    elif args.model == 'add_GP_deriv':
        gppred = fit["gp_pred"]
        Gppred = fit["Gp_pred"]
        fpred_d = fit["f_pred_d"]

        np.save(resultpath+'gppred.npy', gppred)
        np.save(resultpath+'gppred_int.npy', Gppred)
        np.save(resultpath+'fpred_d.npy', fpred_d)

    elif args.model == 'add_GP_second_deriv':
        gppred = fit["gp_pred"]
        Gppred = fit["Gp_pred"]
        GPpred = fit["GP_pred"]

        fpred_d2 = fit["f_pred_d2"]
        fpred_d = fit["f_pred_d"]

        np.save(resultpath+'gppred.npy', gppred)
        np.save(resultpath+'gppred_int.npy', Gppred)
        np.save(resultpath+'gppred_intint.npy', GPpred)

        np.save(resultpath+'fpred_d2.npy', fpred_d2)
        np.save(resultpath+'fpred_d.npy', fpred_d)


    print("Saving done")
