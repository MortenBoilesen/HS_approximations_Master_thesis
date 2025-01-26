import stan
import numpy as np
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pickle
from utils import compute_RMSE, compute_lpd, benchmark_ushaped, compute_rhat


num_chains = 3
num_samples = 100
num_warmup = 200


num_train = 15
num_test = 100
num_pred = 5
b_function_index = 1
std = 1
m = 2
L = 30

''
path = '6_Experiments/63_Ushaped_benchmark_functions/HS_model/'

with open(path + "HSGP_convex_test.stan", "r") as f:
    simulation_code = f.read()


np.random.seed(0)
# Generate data
x_train = np.random.uniform(-5, 5, num_train)
x_test = np.random.uniform(-5, 5, num_test)
y_train = benchmark_ushaped(b_function_index, x_train) + np.random.normal(0, std, num_train)
y_test = benchmark_ushaped(b_function_index, x_test) + np.random.normal(0, std, num_test)
x_pred = np.linspace(-5,5, num_pred)

pred = 5

simulation_data = {
    "predict": pred,
    "num_test": num_test,
    "xtest": x_test,
    "ytest": y_test,

    "num_train": num_train,
    "xtrain": x_train,
    "ytrain": y_train,

    "num_pred": num_pred,
    "xpred": x_pred,

    "jitter": 1e-8,
    "num_basis_functions": m,
    "L": L
}




# Build model
posterior = stan.build(simulation_code, data=simulation_data, random_seed=0)

fit = posterior.sample(num_chains=num_chains, num_samples=num_samples, num_warmup=num_warmup)

print(fit["f_pred"])