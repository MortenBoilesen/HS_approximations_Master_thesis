#%%
import numpy as np
from matplotlib import pyplot as plt

n_samples = 100
X = np.random.uniform(-2.5,2.5,n_samples)
sort_idx = np.argsort(X)
sigma = 1

f = np.where((X >= -2.5) & (X <= -2) | (X >= 2) & (X <= 2.5), 10, 0)
y = f + np.random.normal(0,sigma,f.shape)

num_test = int(0.5*len(X))
num_train = len(X)-num_test

xtest = X[num_train:]
ytest = y[num_train:]

xtrain = X[:num_train]
ytrain = y[:num_train]

plt.plot(xtrain,ytrain, 'o', label='train')
plt.plot(xtest,ytest, 'o', label='test')
plt.plot(X[sort_idx],f[sort_idx],label='f')


#%%
np.save('data/xtrain.npy', xtrain)
np.save('data/ytrain.npy', ytrain)
np.save('data/xtest.npy', xtest)
np.save('data/ytest.npy', ytest)

# %%
