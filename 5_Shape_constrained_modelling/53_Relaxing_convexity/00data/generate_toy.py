#%%
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(1)

path = ''
# path = '6_Experiments/63_Monotonic_data/00data/'

n_samples = 30
sigma = 1

x = np.random.uniform(-5,5,n_samples)
sort_idx = np.argsort(x)
f = 2*np.sin( (x - x.min()) / (x.max() -  x.min())*2*np.pi - np.pi/2 ) +2
# f = np.where( (x <= -2) | (x >= 2) , 5, 0)


# plt.plot(X,f)
y = f + np.random.normal(0,sigma,f.shape)

num_test = int(0.5*len(x))
num_train = len(x)-num_test

xtest = x[num_train:]
ytest = y[num_train:]

xtrain = x[:num_train]
ytrain = y[:num_train]


plt.plot(xtrain,ytrain, 'o', label='train')
plt.plot(xtest,ytest, 'o', label='test')
plt.plot(x[sort_idx],f[sort_idx],label='f')
# plt.savefig(path + 'data.png')
plt.show()
#%%
np.save(path + 'xtrain.npy', xtrain)
np.save(path + 'ytrain.npy', ytrain)
np.save(path + 'xtest.npy', xtest)
np.save(path + 'ytest.npy', ytest)

