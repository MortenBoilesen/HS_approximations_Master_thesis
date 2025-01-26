#%%
import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
def run_experiment(M, T, kernel, y_train, y_test, x_train, x_test, x_pred, benchmark_function, file):
    tf.reset_default_graph()
    np.random.seed(1)
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

    x_pred = x_pred.reshape(-1, 1)
    testsamples = 20
    predsamples = 100

    test_paths = []
    full_paths1 = []

    for s in range(testsamples):
        t_path, t_full_path = sde_solver.forward(
            tf.constant(x_test, dtype=tf.float64),
            save_intermediate=True)
        test_paths.append(t_path)
        full_paths1.append(t_full_path)

    t_test_paths = tf.stack(test_paths)
    t_test_full_paths = tf.stack(full_paths1)

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
    t_pred_full_paths = tf.stack(full_paths2)

    pred_paths = sess.run(t_pred_paths)

    test_mean, test_var = np.mean(pred_paths, axis=0)[:,0], np.var(pred_paths, axis=0)[:,0] + 1 / sess.run(t_beta)
    pred_mean, pred_var = np.mean(pred_paths, axis=0)[:,0], np.var(pred_paths, axis=0)[:,0] + 1 / sess.run(t_beta)

    ftest = test_paths

    yhat = ftest + np.random.normal(0,np.sqrt(1 / sess.run(t_beta)),ftest.shape)

    RMSE_samples = np.sqrt(np.mean((y_test - yhat)**2,axis = 0))
    RMSE_std = np.std(RMSE_samples)
    RMSE = np.mean(RMSE_samples)

    lpd = norm.logpdf(y_test.reshape(-1), loc=test_paths.reshape(testsamples, -1), scale=np.sqrt(1 / sess.run(t_beta)))

    print(f"RMSE: {RMSE:.4f} ± {RMSE_std:.4f}")
    file.write(f"number of inducing points: {M}, simulation time T: {T}, covariance_kernel: {kernel}, RMSE: {RMSE:.3f}, RMSE_std: {RMSE_std:.3f}\n")


    y_test = y_test.reshape(-1)

    ftest = test_paths.reshape(testsamples, -1)
    lpd = norm.logpdf( y_test , loc = test_paths, scale = np.sqrt(1 / sess.run(t_beta)))

    fpred = pred_paths.reshape(predsamples, -1)
    std = np.sqrt(pred_var)

    return RMSE, RMSE_std, M, T, kernel, fpred, ftest, pred_mean, lpd, std



from utils import \
    create_matern32_kernel, \
    create_squared_exp_kernel, \
    real_variable, \
    positive_variable, \
    log_det_from_chol, \
    init_triangular, \
    vec_to_tri

from utils import mu_sigma_tilde, EulerMaruyama, kl_divergence
import argparse
from joblib import Parallel, delayed
parser = argparse.ArgumentParser()
parser.add_argument('--M', type=int, default=40, help='Number of inducing points (default: %(default)s)')
parser.add_argument('--T', type=int, default=1, help='`T` is the total simulation time of the SDE (default: %(default)s)')
parser.add_argument('--num_samples', type=int, default=20, help='number of samples in each chain (default: %(default)s)')
parser.add_argument('--num_warmup', type=int, default=1000 , help='number of samples in warm up (default: %(default)s)')
parser.add_argument('--nu', type=float, default=1.0, help='nu parameter in VP method. (default: %(default)s)')

args = parser.parse_args()

datapath = '6_Experiments/61_Monotonic_benchmark_functions/00data/'
path = '6_Experiments/61_Monotonic_benchmark_functions/SDE_model/'
benchmark_functions = ['flat', 'sinusoidal', 'step', 'linear', 'exp', 'logical']
# benchmark_functions = ['exp']

x_train = np.load(datapath+'x_train.npy')
x_test = np.load(datapath+'x_test.npy')
x_pred = np.linspace( 0, 10, 100)

x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
x_pred = x_pred.reshape(-1,1)

for i in range(len(benchmark_functions)):
    with open(f'6_Experiments/61_Monotonic_benchmark_functions/SDE_model/results/{benchmark_functions[i]}/results.txt', 'w') as file:

        print(f'modelling {benchmark_functions[i]} function')

        file.write(f"model: {benchmark_functions[i]}\n")
        best_RMSE = 100
        best_M = 0
        best_T = 0

        y_train = np.load(datapath+f'{benchmark_functions[i]}/y_train_{benchmark_functions[i]}.npy')
        y_test = np.load(datapath+f'{benchmark_functions[i]}/y_test_{benchmark_functions[i]}.npy')
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        results = Parallel(n_jobs=-1)(delayed(run_experiment)(M, T, kernel, y_train, y_test, x_train, x_test, x_pred, benchmark_functions[i], file)
                              for M in [20, 40]
                              for T in [1, 2, 3, 4, 5]
                              for kernel in ['matern_32', 'sq_exp'])
        
    for RMSE, RMSE_std, M, T, kernel, fpred, ftest, pred_mean, lpd, std in results:
        
        file.write(f"number of inducing points: {M}, simulation time T: {T}, covariance_kernel: {kernel}, RMSE: {RMSE:.3f}, RMSE_std: {RMSE_std:.3f}\n")
        
        if RMSE < best_RMSE:
            best_RMSE = RMSE
            best_M = M
            best_T = T
            best_kernel = kernel

            np.save(f'{path}results/{benchmark_functions[i]}/fpred.npy',fpred)
            np.save(f'{path}results/{benchmark_functions[i]}/ftest.npy',ftest)
            np.save(f'{path}results/{benchmark_functions[i]}/mu.npy',pred_mean)
            np.save(f'{path}results/{benchmark_functions[i]}/std.npy',std)
            np.save(f'{path}results/{benchmark_functions[i]}/lpd.npy',lpd)
                        

        file.write(f"\n Optimization done  with number of inducing points = {best_M}, simulation time T = {best_T}, covariance kernel = {kernel} and RMSE = {best_RMSE} as the best configuration \n")
        print(f"Saving done with number of inducing points = {best_M}, simulation time T = {best_T}, covariance kernel {kernel} and RMSE = {best_RMSE}")