#%%
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import plot_gp_with_samples, plot_with_samples

b_functions = ['flat', 'skew', 'parabola', 'abs', 'sine', 'step']


for b_function in b_functions:

    models = ['baseline_GP', 'HS_model', 'HS_2deriv_model', 'VP_model']
    path = ''
    # path = '6_Experiments/63_Ushaped_benchmark_functions/'

    fig, ax = plt.subplots(1,4, figsize=(20,5))
    i = 0
    model = models[i]
    resultpath = path + model + '/results/' + b_function + '/'

    xtrain = np.load(resultpath + 'xtrain.npy')
    ytrain = np.load(resultpath + 'ytrain.npy')
    xtest = np.load(resultpath + 'xtest.npy')
    ytest = np.load(resultpath + 'ytest.npy')

    xpred = np.linspace( -5, 5, 100)
    fpred = np.load(resultpath + 'fpred.npy')
    mu = np.load(resultpath + 'mu.npy')
    std = np.load(resultpath + 'std.npy')

    plot_gp_with_samples(ax=ax[i], xpred=xpred, fpred=fpred, mu=mu, std=std)
    ax[i].scatter(xtrain,ytrain,label='train')
    ax[i].scatter(xtest,ytest,label='test')
    ax[i].set_title(model.replace('_',' '))

    for i in range(1,4):
        model = models[i]
        resultpath = path + model + '/results/' + b_function + '/'
        with open(resultpath+'model_selection_results.pkl', 'rb') as f:
                model_selection_results = pickle.load(f)

        xtrain = np.load(resultpath + 'test_xtrain.npy')
        ytrain = np.load(resultpath + 'test_ytrain.npy')
        # xtest = np.load(resultpath + 'test_xtest.npy')
        # ytest = np.load(resultpath + 'test_ytest.npy')

        xpred = np.linspace( -5, 5, 100)
        fpred = np.load(resultpath + 'test_fpred.npy')
        sigma = np.load(resultpath + 'test_sigma.npy')

        plot_with_samples(ax=ax[i], xpred = xpred, fpred = fpred, sigma=sigma)
        ax[i].scatter(xtrain,ytrain,label='train')
        # ax[i].scatter(xtest,ytest,label='test')
        title = model.replace('_',' ')
        title += '\n'
        for key, val, in model_selection_results.items():
            title += key + ' : ' + str(val) + '   '
        ax[i].set_title(title)
    
    plt.suptitle(b_function.capitalize() + ' function')

    #%%
        
