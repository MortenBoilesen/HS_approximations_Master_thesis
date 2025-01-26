#%%
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

data = 'fertility_rate'

if data == 'earth_temp':
    df = pd.read_csv('earth_temperature_annual.csv')
    x_samples = (df['Time'].values - np.mean(df['Time'].values))/np.std(df['Time'].values)
    y_samples = df['Anomaly (deg C)'].values
elif data == 'toy':
    x_samples = np.linspace(-1.8,1.8,100)
    f = lambda x: 3*np.exp(x-2) + 0.05*np.sin(5*x)
    y_samples = f(x_samples) + np.random.normal(0,0.25,100)

elif data =='fertility_rate':
    x_samples = np.load('Fertility_rate/orig_data/x_data_year_normal.npy')
    y_samples = np.load('Fertility_rate/orig_data/y_data_normal.npy')


#%%
total_samples = len(x_samples) 
ratio = 0.20
num_test = int(ratio*total_samples)
num_train = total_samples - num_test

train_idx = np.arange(num_train,dtype=int)
test_idx = np.arange(num_train,num_train + num_test, dtype=int)

x_train = x_samples[train_idx]
y_train = y_samples[train_idx]
x_test = x_samples[test_idx]
y_test = y_samples[test_idx]


fig = plt.figure(figsize=(12,10))

colors = ['tab:blue' if i in train_idx else 'tab:orange' for i in range(total_samples)]
plt.scatter(x_samples, y_samples, c=colors)
plt.title(f"{num_train} training points and {num_test} test points.")
train_legend = plt.Line2D([], [], color='tab:blue', marker='o', linestyle='None', markersize=5)
test_legend = plt.Line2D([], [], color='tab:orange', marker='o', linestyle='None', markersize=5)
if data == 'toy':
    plt.plot(x_samples,f(x_samples), color='tab:green', label='True function')
plt.legend([train_legend, test_legend], ['Train', 'Test'])

plt.show()
path = '5_Shape_constrained_modelling/53_Relaxing_convexity/00data/'
path = ''
#%%
if data == 'earth_temp':
    np.save(path + 'earth_temperature_data/x_train.npy',x_train)
    np.save(path + 'earth_temperature_data/y_train.npy', y_train)

    np.save(path + 'earth_temperature_data/x_test.npy', x_test)
    np.save(path + 'earth_temperature_data/y_test.npy', y_test)

elif data == 'fertility_rate':
    np.save(path + 'Fertility_rate/x_train.npy', x_train)
    np.save(path + 'Fertility_rate/y_train.npy', y_train)

    np.save(path + 'Fertility_rate/x_test.npy', x_test)
    np.save(path + 'Fertility_rate/y_test.npy', y_test)



# %%
