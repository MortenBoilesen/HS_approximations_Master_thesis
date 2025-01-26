import numpy as np

data = np.loadtxt('6_Experiments/64_Ushaped_data/all_data/depression_prob_with_age.csv', delimiter=",", skiprows=1)
x_samples = data[:,1]
x_samples = (x_samples - np.mean(x_samples))/np.std(x_samples)
y_samples = data[:,0]*100

np.save('6_Experiments/64_Ushaped_data/all_data/x_samples.npy', x_samples)
np.save('6_Experiments/64_Ushaped_data/all_data/y_samples.npy', y_samples)