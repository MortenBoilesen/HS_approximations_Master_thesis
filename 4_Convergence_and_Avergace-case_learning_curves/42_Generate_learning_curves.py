import numpy as np
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from HSGP import GP_true, squared_exponential, HSGP_box_domain, se_spectral_density_1D
from joblib import Parallel, delayed

sigma = 0.1
kappa = 1
sigmax = 1

def calculate_generalization_error_full(i, X_train, y_train, X_test,f_test):
    GP_approx = GP_true(squared_exponential, theta)
    GP_approx.fit(X_train[i][:, None], y_train[i])
    fhat = GP_approx.predict(X_test[i][:, None])[0]

    return np.mean((fhat - f_test[i])**2)

def calculate_generalization_error_HSGP(i,m, X_train, y_train, X_test, f_test):
    GP_approx = HSGP_box_domain(m, [-10,10], se_spectral_density_1D, theta)
    GP_approx.fit(X_train[i][:, None], y_train[i])
    fhat = GP_approx.predict(X_test[i][:, None])[0]

    return np.mean((fhat - f_test[i])**2)

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--R', type=int, default=10, help='Number of runs for computing std. (default: %(default)s)')
    parser.add_argument('--D', type=int, default=100, help='Number of datasets to average over. (default: %(default)s)')
    parser.add_argument('--S', type=int, default=1000, help='Number of points in test set, (default: %(default)s)')
    parser.add_argument('--n_step', type=int, default=50, help='step in N array. (default: %(default)s)')
    parser.add_argument('--n_jobs', type=int, default=10, help='number of kernels to use. (default: %(default)s)')
    parser.add_argument('--scale', type=float, default=0.05, help='lenghtscale parameter in GP kernel. (default: %(deafult)s)')

    args = parser.parse_args()
    
    scale =args.scale
    scalename = str(scale).replace('.','')
    theta = [sigma, kappa, scale]

    for m in [12,32,64,128,256]:
        print(m)
        nmax = args.S

        N = np.arange(5, nmax, args.n_step)

        Learning_Curve = np.zeros((len(N),args.R,args.D))

        fpath = f'4_Convergence_and_Avergace-case_learning_curves/41_learning_curve_results/f_samples_{scalename}_{args.R}_{args.D}.npy'
        xpath = f'4_Convergence_and_Avergace-case_learning_curves/41_learning_curve_results/x_samples_{scalename}_{args.R}_{args.D}.npy'
        ypath = f'4_Convergence_and_Avergace-case_learning_curves/41_learning_curve_results/y_samples_{scalename}_{args.R}_{args.D}.npy'

        if os.path.exists(fpath) and os.path.exists(xpath) and os.path.exists(ypath):
            X = np.load(xpath)
            f = np.load(fpath)
            y = np.load(ypath)
        else:
            X = np.random.normal(0, sigmax, (args.R, args.D, nmax + args.S))

            GP = GP_true(squared_exponential, theta)
            GP.fit(np.zeros((0,1)), np.zeros((0,1)))

            def generate_samples(r,i,X):
                return GP.generate_samples(X[r,i][:, None], 1).squeeze()

            results = Parallel(n_jobs=args.n_jobs)(delayed(generate_samples)(r,i, X) for r in range(args.R) for i in range(args.D))
            f = np.array(results).reshape(args.R, args.D, nmax + args.S)
            y = f + np.random.normal(0,sigma,f.shape)

            np.save(fpath, f)
            np.save(xpath, X)
            np.save(ypath, y)

        f_test = f[:,:, nmax:]
        X_test = X[:,:, nmax:]


        Generalization_Error = np.zeros((args.R, args.D))

        for k, n in enumerate(N):
            X_train = X[:,:, :n]
            f_train = f[:,:, :n]
            y_train = y[:,:, :n]
            results = Parallel(n_jobs=args.n_jobs)(delayed(calculate_generalization_error_HSGP)(i,m, X_train[r], y_train[r], X_test[r], f_test[r]) for r in range(args.R) for i in range(args.D))            
            results = np.array(results).reshape(args.R,args.D)

            Learning_Curve[k] = results

        np.save(f'4_Convergence_and_Avergace-case_learning_curves/41_learning_curve_results/generalization_error/learning_curve_HSGP_scale={scalename}_D={args.D}_R={args.R}_nstep={args.n_step}_m={m}.npy', Learning_Curve)

    print('full model')
    for k, n in enumerate(N):
        X_train = X[:,:, :n]
        f_train = f[:,:, :n]
        y_train = y[:,:, :n]
        results = Parallel(n_jobs=args.n_jobs)(delayed(calculate_generalization_error_full)(i, X_train[r], y_train[r], X_test[r], f_test[r]) for r in range(args.R) for i in range(args.D))
        results = np.array(results).reshape(args.R,args.D)
        
        Learning_Curve[k] = results

    np.save(f'4_Convergence_and_Avergace-case_learning_curves/41_learning_curve_results/generalization_error/learning_curve_full_scale={scalename}_D={args.D}_nstep={args.n_step}.npy', Learning_Curve)