import numpy as np

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from utils_sde import \
    create_matern32_kernel, \
    create_squared_exp_kernel, \
    real_variable, \
    positive_variable, \
    log_det_from_chol, \
    init_triangular, \
    vec_to_tri

from utils_sde import mu_sigma_tilde, EulerMaruyama, kl_divergence

import time
import sys
import os
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils import compute_RMSE, compute_lpd, benchmark_functions, compute_rhat


def run_simulation(M, T, kernel, seed_0 = 1234):

    print(f'M = {M} and T = {T} and kernel = {kernel}')

    # rhat_test = np.zeros(args.num_experiments, dtype=bool)
    lpd_array = np.zeros(args.num_experiments)
    RMSE_array = np.zeros(args.num_experiments)
    x_pred = np.linspace(0,10,100)

    for i in range(args.num_experiments):
        np.random.seed(i + seed_0)
        # Generate data
        x_train = np.random.uniform(0, 10, args.num_train)
        x_test = np.random.uniform(0, 10, args.num_test)

        y_train = benchmark_functions(args.b_function_index, x_train) + np.random.normal(0, args.std, args.num_train)
        y_test = benchmark_functions(args.b_function_index, x_test) + np.random.normal(0, args.std, args.num_test)

        x_train = x_train.reshape(-1, 1)
        x_test = x_test.reshape(-1, 1)
        x_pred = x_pred.reshape(-1, 1)

        y_train = y_train.reshape(-1,1)

        tf.reset_default_graph()
        tf.set_random_seed(i + seed_0)
        t_pi = tf.constant(np.pi, dtype=tf.float64)

        N_time_steps = 20
        dt = T / N_time_steps
        S = 5
        jitter = 1e-6

        t_X = tf.placeholder(shape=(None, 1), dtype=tf.float64)
        t_Y = tf.placeholder(shape=(None, 1), dtype=tf.float64)

        t_D = tf.shape(t_X)[0]

        t_beta = positive_variable(1.0)
        t_alpha = positive_variable(1.0)
        t_gamma = positive_variable(1.0)
        if kernel == 'sq_exp':
            t_kernel = create_squared_exp_kernel(t_alpha, t_gamma)
        elif kernel == 'matern_32':
            t_kernel = create_matern32_kernel(t_alpha, t_gamma)

        Z_init_space = np.random.uniform(-1, 1, M).reshape(-1, 1)
        t_Z_space = real_variable(Z_init_space)

        Z_init_time = np.random.uniform(-3, 3, M).reshape(-1, 1)
        t_Z_time = T * tf.nn.sigmoid(real_variable(Z_init_time))

        t_Z = tf.concat([t_Z_space, t_Z_time], axis=1)

        t_m = real_variable(1e-1 * np.random.randn(M, 1))
        t_L = vec_to_tri(real_variable(1e-3 * init_triangular(M)), M)
        t_Sigma = tf.matmul(t_L, t_L, transpose_b=True)

        def f(t_space_input, t_time_input):
            return mu_sigma_tilde(t_space_input, t_time_input, t_Z, t_m, t_Sigma, t_kernel)

        sde_solver = EulerMaruyama(f, T, int(T / dt) + 1)

        paths = []
        for s in range(S):
            t_path, _ = sde_solver.forward(t_X, save_intermediate=False)
            paths.append(t_path)

        t_paths = tf.stack(paths)

        t_first_term = -(t_D / 2) * tf.log(2 * t_pi / t_beta) \
                    -(t_beta / 2) * tf.reduce_sum((t_paths - t_Y)**2, axis=(1,2))
        t_first_term = tf.reduce_mean(t_first_term)

        t_K_ZZ = t_kernel.covar_matrix(t_Z, t_Z)
        t_L_Z = tf.cholesky(t_K_ZZ + jitter * tf.diag(tf.ones(M, dtype=tf.float64)))
        t_second_term = kl_divergence(t_L_Z, t_m, t_Sigma)

        t_lower_bound = t_first_term - t_second_term
        t_neg_lower_bound = -tf.reduce_sum(t_lower_bound)

        t_lr = tf.placeholder(dtype=tf.float64)
        optimiser = tf.train.AdamOptimizer(learning_rate=t_lr).minimize(t_neg_lower_bound)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        n_steps = 10000
        lr = 1e-2
        best_loss = np.inf

        feed_dict = {t_Y: y_train, t_X: x_train, t_lr: lr}

        for step in range(n_steps + 1):
            _, loss = sess.run([optimiser, t_neg_lower_bound], feed_dict)
            
            if step % 100 == 0:
                print('iter {:5}: {:10.3f}, {:.2e}'.format(step, loss, np.sqrt(1 / sess.run(t_beta))))
                if loss < best_loss:
                    best_loss = loss
                    saver.save(sess, 'checkpoints/monotonic_model')

        saver.restore(sess, 'checkpoints/monotonic_model')

        testsamples = args.num_samples
        predsamples = 500

        test_paths = []
        full_paths1 = []

        for s in range(testsamples):
            t_path, t_full_path = sde_solver.forward(
                tf.constant(x_test, dtype=tf.float64),
                save_intermediate=True)
            test_paths.append(t_path)
            full_paths1.append(t_full_path)

        t_test_paths = tf.stack(test_paths)

        test_paths = sess.run(t_test_paths)

        if i == args.num_experiments - 1:
            
            pred_paths = []
            full_paths2 = []

            for s in range(predsamples):
                t_path, t_full_path = sde_solver.forward(
                    tf.constant(x_pred, dtype=tf.float64),
                    save_intermediate=True)
                pred_paths.append(t_path)
                full_paths2.append(t_full_path)

            t_pred_paths = tf.stack(pred_paths)

            pred_paths = sess.run(t_pred_paths)
            fpred = pred_paths[:,:,0].T
        
        sigma = np.sqrt(1 / sess.run(t_beta))

        f_test = test_paths[:,:,0].T
        lpd_array[i] = compute_lpd(y_test, f_test, sigma)
        RMSE_array[i] = compute_RMSE(y_test, f_test)

    lpd_mean = np.mean(lpd_array)
    RMSE_mean = np.mean(RMSE_array)
    lpd_std = np.std(lpd_array)
    RMSE_std = np.std(RMSE_array)

    return RMSE_mean, RMSE_std, lpd_mean, lpd_std, x_train, y_train, fpred, sigma

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train', type=int, default=15, help='number of training points (default: %(default)s)')
    parser.add_argument('--num_test', type=int, default=100, help='number of test points (default: %(default)s)')
    parser.add_argument('--b_function_index', type=int, default=0, help='index of benchmark function (default: %(default)s)')
    parser.add_argument('--std', type=float, default=1, help='std of noise added to the training and test data (default: %(default)s)')
    parser.add_argument('--num_experiments', type=int, default=20, help='number of experiments (default: %(default)s)')
    parser.add_argument('--num_samples', type=int, default=500, help='number of samples in each chain (default: %(default)s)')
    parser.add_argument('--num_warmup', type=int, default=200, help='number of samples in each chain (default: %(default)s)')

    args = parser.parse_args()

    datapath = '6_Experiments/61_Monotonic_benchmark_functions/00data/'
    path = '6_Experiments/61_Monotonic_benchmark_functions/SDE_model/'
    benchmark_function_names = ['flat', 'sinusoidal', 'step', 'linear', 'exp', 'logical']
    benchmark_function = benchmark_function_names[args.b_function_index]

    resultpath = path + 'results/' + benchmark_function +'/'

    os.makedirs(resultpath, exist_ok = True)

    with open(resultpath+'model_selection_results.pkl', 'rb') as file:
        best_configuration = pickle.load(file)


    print(best_configuration)

    RMSE_mean, RMSE_std, lpd_mean, lpd_std, x_train, y_train, fpred, sigma = run_simulation(best_configuration['M'], best_configuration['T'], best_configuration['kernel'], seed_0 = 1234*(args.b_function_index + 1))
    best_configuration_results = np.zeros(4)
    best_configuration_results[0] = RMSE_mean
    best_configuration_results[1] = RMSE_std
    best_configuration_results[2] = lpd_mean
    best_configuration_results[3] = lpd_std
    
    np.save(resultpath+'10000_test_results.npy', (best_configuration_results))
    np.save(resultpath+'10000_test_xtrain.npy', (x_train))
    np.save(resultpath+'10000_test_ytrain.npy', (y_train))
    np.save(resultpath+'10000_test_fpred.npy', (fpred))
    np.save(resultpath+'10000_test_sigma.npy', (sigma))