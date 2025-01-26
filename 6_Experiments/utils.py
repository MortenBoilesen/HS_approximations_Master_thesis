
#%%
import numpy as np
import scipy.stats as sp
import scipy.special as ss
#%%
# b_function = 'exp'
# path = '6_Experiments/61_Monotonic_benchmark_functions/HS_model/'
# datapath = '6_Experiments/61_Monotonic_benchmark_functions/00data/'

# f_test = np.load(path + f'results/{b_function}/fpred.npy')
# sigma = np.load(path + f'results/{b_function}/sigma.npy')
# y_test = np.load(datapath + f'{b_function}/y_test_{b_function}.npy')



# %%
def compute_RMSE(y_test, f_test):
    y_hat = np.mean(f_test, axis=1)
    return np.sqrt(np.sum((y_test - y_hat)**2) / len(y_test))

def compute_lpd(y_test, f_test, sigma):
    return np.mean(ss.logsumexp(a=sp.norm.logpdf(y_test[:,None], loc=f_test, scale=sigma), b=1/f_test.shape[1], axis=1))

def benchmark_functions(idx, x):
    if idx == 0:
        return 3*np.ones_like(x)
    elif idx == 1:
        return 0.32*(x + np.sin(x))
    elif idx == 2:
        return 3+3*(np.heaviside(x-5,1))
    elif idx == 3:
        return 0.3*x
    elif idx == 4:
        return 0.15*np.exp(0.6*x-3)
    elif idx == 5:
        return 3/(1+np.exp(-2*x+10))

def benchmark_ushaped(idx, x):
    if idx == 0:
        return 2*np.ones(x.shape)
    elif idx == 1:
        return 0.1*(x-2)**2
    elif idx == 2:
        return 0.25*x**2
    elif idx == 3:
        return np.abs(x)
    elif idx == 4:
        return -2*np.sin( (x - x.min()) / (x.max() -  x.min())*2*np.pi - np.pi/2 ) +2
    elif idx == 5:
        y = 4*np.ones(np.shape(x))
        y[np.abs(x) <= 3] = 0
        return y

def compute_rhat(parameter_array, num_chains, num_samples):
    '''
        parameter_array (num_parameters x num_saples * chains )

        '''
    rhats = np.zeros((num_chains,len(parameter_array)))
    for idx_chain in range(num_chains):
        sub_chains = []
        half_num_samples = int(0.5*num_samples)
        idx = np.arange(idx_chain,num_chains*num_samples,num_chains)
        chain = (parameter_array.T)[idx]
        sub_chains.append(chain[:half_num_samples, :])
        sub_chains.append(chain[half_num_samples:, :])

        # count number of sub chains
        num_sub_chains = len(sub_chains)
            
        # compute mean and variance of each subchain
        chain_means = np.array([np.mean(s, axis=0) for s in sub_chains])                                             # dim: num_sub_chains x num_params
        chain_vars = np.array([1/(num_samples-1)*np.sum((s-m)**2, 0) for (s, m) in zip(sub_chains, chain_means)])    # dim: num_sub_chains x num_params

        # compute between chain variance
        global_mean = np.mean(chain_means, axis=0)                                                                   # dim: num_params
        B = num_samples/(num_sub_chains-1)*np.sum((chain_means - global_mean)**2, axis=0)                            # dim: num_params

        # compute within chain variance
        W = np.mean(chain_vars, 0)                                                                                   # dim: num_params                                                          

        # compute estimator and return
        var_estimator = (num_samples-1)/num_samples*W + (1/num_samples)*B                                            # dim: num_params 
        rhats[idx_chain] = np.sqrt(var_estimator/W)

    return rhats

def compute_rhat_across_chains(parameter_array, num_chains, num_samples):
    rhats = np.zeros(len(parameter_array))
    sub_chains = []

    for idx_chain in range(num_chains):
        half_num_samples = int(0.5*num_samples)
        idx = np.arange(idx_chain,num_chains*num_samples,num_chains)
        chain = (parameter_array.T)[idx]
        sub_chains.append(chain[:half_num_samples, :])
        sub_chains.append(chain[half_num_samples:, :])

    # count number of sub chains
    num_sub_chains = len(sub_chains)
        
    # compute mean and variance of each subchain
    chain_means = np.array([np.mean(s, axis=0) for s in sub_chains])                                             # dim: num_sub_chains x num_params
    chain_vars = np.array([1/(num_samples-1)*np.sum((s-m)**2, 0) for (s, m) in zip(sub_chains, chain_means)])    # dim: num_sub_chains x num_params

    # compute between chain variance
    global_mean = np.mean(chain_means, axis=0)                                                                   # dim: num_params
    B = num_samples/(num_sub_chains-1)*np.sum((chain_means - global_mean)**2, axis=0)                            # dim: num_params

    # compute within chain variance
    W = np.mean(chain_vars, 0)                                                                                   # dim: num_params                                                          

    # compute estimator and return
    var_estimator = (num_samples-1)/num_samples*W + (1/num_samples)*B                                            # dim: num_params 
    rhats = np.sqrt(var_estimator/W)

    return rhats


def plot_with_samples(ax, xpred, fpred, sigma, distribution = 'predective', percentile=95, color='r',alpha=0.2, color_samples='g', color_CI = 'tab:green', alpha_samples=0.2, title="", num_samples=0):
    
    mu = np.mean(fpred, axis = 1)
    if distribution == 'predective':
        ypred = fpred + sigma*np.random.normal(0,1,fpred.shape)
        upper = np.percentile(ypred, percentile, axis=1)
        lower = np.percentile(ypred, 100-percentile, axis=1)
    elif distribution == 'posterior':
        upper = np.percentile(fpred, percentile, axis=1)
        lower = np.percentile(fpred, 100-percentile, axis=1)

    
    # plot distribution
    ax.fill_between(xpred, lower, upper, color=color_CI, alpha=alpha, label=f'{percentile}% {distribution} interval')
    
    # plot samples
    if num_samples > 0: 
        index = np.random.choice(fpred.shape[1], num_samples)
        fs = fpred[:,index]
        ax.plot(xpred, fs[:,0], color=color_samples, alpha=alpha_samples, label="$f(x)$ samples")
        ax.plot(xpred, fs[:, 1:], color=color_samples, alpha=alpha_samples)

    ax.plot(xpred, mu, color=color, label='Posterior mean', linewidth=2)
    ax.plot(xpred, upper, color=color, linestyle='--')
    ax.plot(xpred, lower, color=color, linestyle='--')

def plot_gp_with_samples(ax, xpred, fpred, mu, std, color='r', color_samples='g', alpha_samples=0.2, title="", num_samples=0):
    ax.fill_between(xpred.ravel(), mu - 2*std, mu + 2*std, color='tab:green', alpha=0.2, label='95% interval')
    
    if num_samples > 0: 
        index = np.random.choice(fpred.shape[1], num_samples)
        fs = fpred[:,index]
        ax.plot(xpred, fs[:,0], color=color_samples, alpha=alpha_samples, label="$f(x)$ samples")
        ax.plot(xpred, fs[:, 1:], color=color_samples, alpha=alpha_samples)

    ax.plot(xpred, mu, color='r', label='Posterior mean', linewidth = 2)
    ax.plot(xpred, mu + 2*std, color='r', linestyle='--')
    ax.plot(xpred, mu - 2*std, color='r', linestyle='--')


def extract_chains(stan_output, num_chains, num_samples):
    chains = []
    for idx_chain in range(num_chains):
        idx = np.arange(idx_chain,num_chains*num_samples,num_chains)
        chains.append((stan_output.T)[idx])

    return np.stack(chains, axis=0)

def gelman_rubin(x):
    """ Estimate the marginal posterior variance. Vectorised implementation. """
    m_chains, n_iters = x.shape

    # Calculate between-chain variance
    B_over_n = ((np.mean(x, axis=1) - np.mean(x))**2).sum() / (m_chains - 1)

    # Calculate within-chain variances
    W = ((x - x.mean(axis=1, keepdims=True))**2).sum() / (m_chains*(n_iters - 1))

    # (over) estimate of variance
    s2 = W * (n_iters - 1) / n_iters + B_over_n

    return s2


def compute_effective_sample_size_single_param(x):
    """ Compute the effective sample size of estimand of interest. Vectorised implementation. """
    m_chains, n_iters = x.shape

    variogram = lambda t: ((x[:, t:] - x[:, :(n_iters - t)])**2).sum() / (m_chains * (n_iters - t))

    post_var = gelman_rubin(x)

    t = 1
    rho = np.ones(n_iters)
    negative_autocorr = False

    # Iterate until the sum of consecutive estimates of autocorrelation is negative
    while not negative_autocorr and (t < n_iters):
        rho[t] = 1 - variogram(t) / (2 * post_var)

        if not t % 2:
            negative_autocorr = sum(rho[t-1:t+1]) < 0

        t += 1

    return int(m_chains*n_iters / (1 + 2*rho[1:t].sum()))

def compute_effective_sample_size(chains):
    """ computes the effective sample size for each parameter in a MCMC simulation. 
        The function expects the argument chain to be a numpy array of shape (num_chains x num_samples x num_params)
        and it return a numpy of shape (num_params) containing the S_eff estimates for each parameter
    """
    # get dimensions
    num_chains, num_samples, num_params = chains.shape

    # estimate sample size for each parameter
    S_eff = np.array([compute_effective_sample_size_single_param(chains[:, :, idx_param]) for idx_param in range(num_params)])

    # return
    return S_eff
