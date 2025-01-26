#%%
import numpy as np
from matplotlib import pyplot as plt
import os
np.random.seed(1234)

if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--R', type=int, default=2, help='What run (default: %(default)s)')
    args = parser.parse_args()

    np.random.seed(100 + args.R)

    x_samples = np.load('6_Experiments/64_Ushaped_data/all_data/x_samples.npy')
    y_samples = np.load('6_Experiments/64_Ushaped_data/all_data/y_samples.npy')
        
    total_samples = len(x_samples) 
    ratio = 0.25
    num_test = int(ratio*total_samples)
    num_train = total_samples - num_test

    train_idx = np.random.choice(np.arange(total_samples,dtype=int), num_train, replace=False)
    test_idx = np.setdiff1d(np.arange(total_samples), train_idx)

    x_train = x_samples[train_idx]
    y_train = y_samples[train_idx]
    x_test = x_samples[test_idx]
    y_test = y_samples[test_idx]

    np.random.shuffle(train_idx)
    x_train_shuffled = x_samples[train_idx]
    y_train_shuffled = y_samples[train_idx]


    path = f'6_Experiments/64_Ushaped_data/Runs/Run_{args.R}/00data/'
    os.makedirs(path, exist_ok=True)

    np.save(path + 'x_train.npy', x_train)
    np.save(path + 'y_train.npy', y_train)
    np.save(path + 'x_train_shuffled.npy', x_train)
    np.save(path + 'y_train_shuffled.npy', y_train)
    np.save(path + 'x_test.npy', x_test)
    np.save(path + 'y_test.npy', y_test)
