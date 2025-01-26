#%%
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

import sys
import os
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils import compute_RMSE, compute_lpd, benchmark_functions, compute_rhat


def run_simulation(M, T, kernel, x_train, x_test, x_pred, y_train, y_test):

    print(f'M = {M} and T = {T} and kernel = {kernel}')


    np.random.seed(1)
    # Generate data
    x_train = x_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)
    x_pred = x_pred.reshape(-1,1)

    y_train = -y_train.reshape(-1,1)
    y_test = -y_test

    tf.reset_default_graph()
    tf.set_random_seed(1)
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

    n_steps = 3000
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
    predsamples = 200

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
    
    sigma = np.sqrt(1 / sess.run(t_beta))

    f_test = test_paths[:,:,0].T
    f_pred = pred_paths[:,:,0].T
    lpd_array = compute_lpd(y_test, f_test, sigma)
    RMSE_array = compute_RMSE(y_test, f_test)

    return lpd_array, RMSE_array, f_pred, sigma, f_test
#%%
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train', type=int, default=15, help='number of training points (default: %(default)s)')
    parser.add_argument('--num_test', type=int, default=100, help='number of test points (default: %(default)s)')
    parser.add_argument('--num_samples', type=int, default=500, help='number of samples in each chain (default: %(default)s)')

    args = parser.parse_args()

    path = '5_Shape_constrained_modelling/53_Relaxing_convexity/SDE_model/'
    # path = ''
    datapath = '5_Shape_constrained_modelling/53_Relaxing_convexity/00data/'
    # datapath = '../00data/'
    resultpath = path + 'results/'

    os.makedirs(resultpath, exist_ok = True)

    x_train = np.load(datapath + 'Fertility_rate/x_train.npy')
    y_train = np.load(datapath + 'Fertility_rate/y_train.npy')
    x_test = np.load(datapath + 'Fertility_rate/x_test.npy')
    y_test = np.load(datapath + 'Fertility_rate/y_test.npy')
    x_pred = np.linspace(np.min(x_train), np.max(x_test), 100)

    with open(resultpath+'model_selection_results.pkl', 'rb') as file:
        best_configuration = pickle.load(file)

    lpd_array, RMSE_array, f_pred, sigma, f_test = run_simulation(best_configuration['M'], best_configuration['T'], best_configuration['kernel'], x_train, x_test, x_pred, y_train, y_test)

    best_configuration_results = np.zeros(2)
    best_configuration_results[0] = lpd_array
    best_configuration_results[1] = RMSE_array

    # Save the best results
    np.save(resultpath+'test_results.npy', (best_configuration_results))
    np.save(resultpath+'test_fpred.npy', -f_pred)
    np.save(resultpath+'test_sigma.npy', sigma)
    np.save(resultpath+'test_ftest.npy', -f_test)

#%%

# test_results = np.load(resultpath+'test_results.npy')
# %%
