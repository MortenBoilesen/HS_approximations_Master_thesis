#%%
import numpy as np
from scipy.stats import norm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import compute_RMSE, compute_lpd

import matplotlib.pyplot as plt

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from utils import \
    create_matern32_kernel, \
    create_squared_exp_kernel, \
    real_variable, \
    positive_variable, \
    log_det_from_chol, \
    init_triangular, \
    vec_to_tri

from 5_Shape_constrained_modelling/53_Relaxing_convexity.SDE_model.utils_sde import mu_sigma_tilde, EulerMaruyama, kl_divergence

#%%
tf.reset_default_graph()
np.random.seed(1)
tf.set_random_seed(1)
t_pi = tf.constant(np.pi, dtype=tf.float64) # to make dtype = tf.float64


#Parameters of the monotonic flow model:
# `M` is the number of inducing points for the flow GP (`g(s, t)` in the paper),
# `T` is the total simulation time,
# `N_time_steps` is the number of steps in the numerical SDE solver. The step size `dt` is `T / N_time_steps`
#  `S` is the number of sampled trajectories used to estimate the expectation in the ELBO

M = 20
T = 1
N_time_steps = 20
dt = T / N_time_steps
S = 5
jitter = 1e-6

#Define placeholders for 1D inputs and observations
t_X = tf.placeholder(shape=(None, 1), dtype=tf.float64)
t_Y = tf.placeholder(shape=(None, 1), dtype=tf.float64)

t_D = tf.shape(t_X)[0] # Number of inputs to be determined at runtime

#Define hyperparameters

t_beta = positive_variable(1.0)

t_alpha = positive_variable(1.0)
t_gamma = positive_variable(1.0)
t_kernel = create_squared_exp_kernel(t_alpha, t_gamma)

#Define variables for the inducing locations (in space and time for the flow GP):

# Inducing locations in space initialised between -1 and 1
Z_init_space = np.random.uniform(-1, 1, M).reshape(-1, 1)
t_Z_space = real_variable(Z_init_space)

# Inducing locations in time are constrained to be between 0 and T
# by applying the sigmoid function on the corresponding variable
Z_init_time = np.random.uniform(-3, 3, M).reshape(-1, 1)
t_Z_time = T * tf.nn.sigmoid(real_variable(Z_init_time))

# Concatenate space and time inducing locations into a single tensor
t_Z = tf.concat([t_Z_space, t_Z_time], axis=1)


#Define variable for the Variational distribution
t_m = real_variable(1e-1 * np.random.randn(M, 1))

t_L = vec_to_tri(real_variable(1e-3 * init_triangular(M)), M) # Cholesky factor
t_Sigma = tf.matmul(t_L, t_L, transpose_b=True)


#%%
# We define a function computing the parameters of the flow GP posterior distribution at a given spatio-temporal point (equation (2) in the paper): 


def f(t_space_input, t_time_input):
    return mu_sigma_tilde(t_space_input, t_time_input, t_Z, t_m, t_Sigma, t_kernel)

sde_solver = EulerMaruyama(f, T, int(T / dt) + 1) # jointly_gaussian=True

#%%
#We use the SDE solver to sample `S` outputs of the monotonic flow:
paths = []
for s in range(S):
    t_path, _ = sde_solver.forward(t_X, save_intermediate=False)
    paths.append(t_path)

t_paths = tf.stack(paths)

#%%
#Estimating the ELbo (eq. 4)

t_first_term = -(t_D / 2) * tf.log(2 * t_pi / t_beta) \
               -(t_beta / 2) * tf.reduce_sum((t_paths - t_Y)**2, axis=(1,2))
t_first_term = tf.reduce_mean(t_first_term)

t_K_ZZ = t_kernel.covar_matrix(t_Z, t_Z)
t_L_Z = tf.cholesky(t_K_ZZ + jitter * tf.diag(tf.ones(M, dtype=tf.float64)))
t_second_term = kl_divergence(t_L_Z, t_m, t_Sigma)

t_lower_bound = t_first_term - t_second_term
t_neg_lower_bound = -tf.reduce_sum(t_lower_bound)

#%%
# Load world temperature data
x_train = np.load('../00data/Fertility_rate/x_train.npy')
y_train = np.load('../00data/Fertility_rate/y_train.npy')

x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)

x_test = np.load('../00data/Fertility_rate/x_test.npy')
y_test = np.load('../00data/Fertility_rate/y_test.npy')

y_train = -y_train
y_test = -y_test

y_test = (y_test - np.mean(y_train))/np.std(y_train)

x_test = x_test.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


fig = plt.figure(figsize=(12,10))
plt.scatter(x_train.ravel(), y_train.ravel(), color='blue')
plt.scatter(x_test.ravel(), y_test.ravel(), color='orange')
plt.show()

#%%
#Set up the optimiser and initialise the variables
t_lr = tf.placeholder(dtype=tf.float64)
optimiser = tf.train.AdamOptimizer(learning_rate=t_lr).minimize(t_neg_lower_bound)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

#%%
#Run the optimisation

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

# Load the best model
saver.restore(sess, 'checkpoints/monotonic_model')

#%% 
#Visualise the fitted model


x_pred = np.linspace( np.min(x_train), np.max(x_test) , 100 ).reshape(-1, 1)
testsamples = 300
predsamples = 100

test_paths = []
full_paths1 = []


print('first sampling round')
for s in range(testsamples):
    t_path, t_full_path = sde_solver.forward(
        tf.constant(x_test, dtype=tf.float64),
        save_intermediate=True)
    test_paths.append(t_path)
    full_paths1.append(t_full_path)

    if s % (testsamples // 10) == 0:
        print(f'{(s / testsamples) * 100:.0f}% complete')

t_test_paths = tf.stack(test_paths)
t_test_full_paths = tf.stack(full_paths1)

test_paths = sess.run(t_test_paths)


pred_paths = []
full_paths2 = []

print('second sampling round')
for s in range(predsamples):
    t_path, t_full_path = sde_solver.forward(
        tf.constant(x_pred, dtype=tf.float64),
        save_intermediate=True)
    pred_paths.append(t_path)
    full_paths2.append(t_full_path)

    if s % (predsamples // 10) == 0:
        print(f'{(s / predsamples) * 100:.0f}% complete')

t_pred_paths = tf.stack(pred_paths)
t_pred_full_paths = tf.stack(full_paths2)

pred_paths = sess.run(t_pred_paths)


test_mean, test_var = np.mean(pred_paths, axis=0)[:,0], np.var(pred_paths, axis=0)[:,0] + 1 / sess.run(t_beta)
pred_mean, pred_var = np.mean(pred_paths, axis=0)[:,0], np.var(pred_paths, axis=0)[:,0] + 1 / sess.run(t_beta)

#%%

plt.figure(figsize=(12,10))
plt.scatter(x_train, y_train)
plt.scatter(x_train, y_train)
plt.plot(x_pred, pred_mean, color='red', label='Posterior mean')
plt.show()

#%%
#save the results
sigma = np.sqrt(1 / sess.run(t_beta))
y_test = y_test.reshape(-1)
f_test = test_paths.reshape(testsamples, -1).T
lpd = compute_lpd(y_test,f_test, sigma)
rmse = compute_RMSE(y_test, f_test)

f_pred = pred_paths.reshape(predsamples, -1).T
std = np.sqrt(pred_var + sigma*sigma)


f_pred = -f_pred
pred_mean = -pred_mean

np.save('results/fpred.npy', f_pred)
np.save('results/mu.npy', pred_mean)
np.save('results/sigma.npy', np.array([sigma]))
np.save('results/std.npy',std)
np.save('results/lpd.npy',lpd)
np.save('results/rmse.npy',rmse)
# %%
