#%%
import numpy as np
from scipy.stats import norm

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

import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils import compute_RMSE, compute_lpd, benchmark_functions, compute_rhat

def run_simulation(M, T, kernel, x_data, y_data):


    # rhat_test = np.zeros(args.num_experiments, dtype=bool)
    lpd_array = np.zeros(args.num_experiments)
    RMSE_array = np.zeros(args.num_experiments)

    for i in range(0,args.num_experiments):
        np.random.seed(i)
        # Generate data
        x_train = x_data[i*2:36 + i*2]
        x_test = x_data[36 + i*2:42 + i*2]
        y_train =  y_data[i*2:36 + i*2]
        y_test =  y_data[36 + i*2:42 + i*2]

        x_train = x_train.reshape(-1, 1)
        x_test = x_test.reshape(-1, 1)

        y_train = -y_train.reshape(-1,1)
        y_test = -y_test

        tf.reset_default_graph()
        tf.set_random_seed(i)
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

        n_steps = 2000
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
        
        sigma = np.sqrt(1 / sess.run(t_beta))

        ftest = test_paths[:,:,0].T
        lpd_array[i-1] = compute_lpd(y_test, ftest, sigma)
        RMSE_array[i-1] = compute_RMSE(y_test, ftest)

    lpd_mean = np.mean(lpd_array)
    RMSE_mean = np.mean(RMSE_array)
    lpd_std = np.std(lpd_array)
    RMSE_std = np.std(RMSE_array)

    return RMSE_mean, RMSE_std, lpd_mean, lpd_std, RMSE_array, lpd_array



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_experiments', type=int, default=5, help='number of experiments (default: %(default)s)')
    parser.add_argument('--num_samples', type=int, default=500, help='number of samples in each chain (default: %(default)s)')
    parser.add_argument('--num_warmup', type=int, default=500, help='number of samples in warm up (default: %(default)s)')
    
    args = parser.parse_args()

    path = '5_Shape_constrained_modelling/53_Relaxing_convexity/SDE_model/'
    datapath = '5_Shape_constrained_modelling/53_Relaxing_convexity/00data/'
    resultpath = path + 'results/'
    samplepath = resultpath + 'experiment_samples/'

    os.makedirs(resultpath, exist_ok = True)
    os.makedirs(samplepath, exist_ok = True)

    M_list = [20, 40]
    T_list = [1,2,3,4,5]
    kernel_list = ['matern_32', 'sq_exp']
    best_M = 100
    best_T = 100
    best_kernel = 'None'
    best_RMSE = 1000

    #load data
    x_data = np.load(datapath + 'Fertility_rate/x_train.npy')
    y_data = np.load(datapath + 'Fertility_rate/y_train.npy')

    with open(f'{path}/results/model_selection_results.txt', 'w') as file:
        file.write(f"Forecasting model selection\n")

    for M in M_list:
        for T in T_list:
            for kernel in kernel_list:
                print(f'M = {M}, T = {T} and kernel = {kernel}')
                result = run_simulation(M, T, kernel, x_data, y_data)
                print(f'Experiment done for M = {M} and T = {T} and kernel = {kernel}')
                RMSE_mean, RMSE_std, lpd_mean, lpd_std, RMSE_array, lpd_array = result

                with open(f'{path}/results/model_selection_results.txt', 'a') as file:
                    file.write(f"M: {M}, T: {T}, kernel: {kernel}, RMSE: {RMSE_mean:.3f}, RMSE_std: {RMSE_std:.3f}, lpd: {lpd_mean:.3f}, lpd_std: {lpd_std:.3f}\n")

                np.save(samplepath + f'RMSE_M_{M}_T_{T}_{kernel}.npy', RMSE_array)
                np.save(samplepath + f'lpd_M_{M}_T_{T}_{kernel}.npy', lpd_array)
                
                if RMSE_mean < best_RMSE:
                    best_RMSE = RMSE_mean
                    best_M = M
                    best_T = T
                    best_kernel = kernel

    # Save the best results
    best_configuration = {'M': best_M, 'T': best_T, 'kernel': best_kernel} 

    with open(f'{path}results/model_selection_results.pkl', 'wb') as file:
        pickle.dump(best_configuration, file)