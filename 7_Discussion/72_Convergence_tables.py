#%%
import numpy as np
import pandas as pd
from utils import compute_rhat_across_chains, extract_chains, compute_effective_sample_size
paths = ['5_Shape_constrained_modelling/53_Relaxing_convexity', '6_Experiments/61_Monotonic_benchmark_functions', '6_Experiments/63_Ushaped_benchmark_functions']

num_chains = 4
num_samples = 10000

############################### RHAT PR. CHAIN #########################
print('Individual split-Rhat')
# MONOTONIC
print('\tMonotonic')
columns = pd.MultiIndex.from_tuples([   ("Monotonic model" , "data"),
                                        ("Monotonic model" , "converged parameters"),
                                        ("Monotonic model" , "converged chains"),                                
                                        ("Monotonic model" , "converged runs"),
                                       ])

df_convergence = pd.DataFrame(columns=columns)

parameter_convergence = []
parameter_effective_samples = []
chain_convergence = []
run_convergence = []
run_effective_samples = []

data_monotonic = ['flat', 'sinusoidal', 'step', 'linear', 'exp', 'logical', 'fertility']


for data in data_monotonic:
    if data != 'fertility':
        path = '6_Experiments/61_Monotonic_benchmark_functions'
        # Asses rhat
        rhat = np.load(path+f'/HS_model/results/{data}/params_and_rhat/10000_test_rhat.npy')
        num_params = np.prod(rhat.shape)
        parameter_convergence.append(f'{np.sum(rhat <= 1.01)}/{num_params}')
        chains_converged = np.prod(rhat <= 1.01, axis=-1)
        chain_convergence.append(f'{np.sum(chains_converged)}/{num_chains*20}')
        runs_converged = np.prod(chains_converged, axis=-1)
        run_convergence.append(f'{np.sum(runs_converged)}/20')
    
    else:
        path = '5_Shape_constrained_modelling/53_Relaxing_convexity'
        rhat = np.load(path + f'/HS_model/results/params_and_rhat/10000_test_rhat.npy')
        parameter_convergence.append(f'{np.sum(rhat <= 1.01)}/{np.prod(rhat.shape)}')
        chains_converged = np.prod(rhat <= 1.01, axis=-1)
        chain_convergence.append(f'{np.sum(chains_converged)}/{num_chains}')
        run_convergence.append('na')


df_convergence.loc[:,  ("Monotonic model" , "data")] = data_monotonic
df_convergence.loc[:,  ("Monotonic model" , "converged parameters")] = parameter_convergence
df_convergence.loc[:,  ("Monotonic model" , "converged chains")] = chain_convergence
df_convergence.loc[:,  ("Monotonic model" , "converged runs")] = run_convergence

print(df_convergence)
latex = df_convergence.to_latex(index = False, column_format='lrrr', multicolumn_format='c')
with open('7_Discussion/exp1/convergence_table_monotonic.tex', 'w') as f:
    f.write(latex)
print()
# print(df_convergence.to_latex(index = False, column_format='lrrr', multicolumn_format='c'))
#%%

## USHAPED
print('\tUshaped')
columns = pd.MultiIndex.from_tuples([   ("Ushaped model" , "data"),
                                        ("Ushaped model" , "converged parameters"),
                                        ("Ushaped model" , "converged chains"),
                                        ("Ushaped model" , "converged runs"),
                                       ])

df_convergence = pd.DataFrame(columns=columns)
parameter_convergence = []
chain_convergence = []
run_convergence = []


data_ushaped = ['flat', 'skew', 'parabola', 'abs', 'sine', 'step']
fractions = [0.5, 0.7272727272727273, 0.8888888888888888, 1.0, 2.0, 4.0]

for data in data_ushaped:
    if data != 'depression':
        path = '6_Experiments/63_Ushaped_benchmark_functions'
        rhat = np.load(path+f'/HS_model/results/{data}/params_and_rhat/10000_test_rhat_array.npy')
        parameter_convergence.append(f'{np.sum(rhat <= 1.01)}/{np.prod(rhat.shape)}')
        chains_converged = np.prod(rhat <= 1.01, axis=-1)
        chain_convergence.append(f'{np.sum(chains_converged)}/{num_chains*20}')
        runs_converged = np.prod(chains_converged, axis=-1)
        run_convergence.append(f'{np.sum(runs_converged)}/20')
    
        # if data == 'skew':
        #     print(rhat.shape)
        #     for i in range(len(rhat)):
        #         print(f'{i}: ', np.sum(rhat[i] <= 1.01, axis=1 ))
            

path = '6_Experiments/64_Ushaped_data'
for fraction in fractions:
    parameters_converged = 0
    parameters_total = 0
    chains_converged = 0
    runs_converged = 0

    for r in range(4):
        rhat = np.load(path + f'/Runs/Run_{r+1}/HS_model/{fraction}/params_and_rhat/test_rhat.npy')
        parameters_converged += np.sum(rhat <= 1.01)
        parameters_total += rhat.shape[1]*num_chains
        chains_converged += np.sum(np.prod(rhat <= 1.01, axis=1))
        runs_converged += np.prod(rhat <= 1.01)

    
    parameter_convergence.append(f'{parameters_converged}/{parameters_total}')
    chain_convergence.append(f'{np.sum(chains_converged)}/{num_chains*7}')
    run_convergence.append(f'{np.sum(runs_converged)}/7')
        
        
data_ushaped += [f'depression {int(fraction*42//8)}' for fraction in fractions]
df_convergence.loc[:,  ("Ushaped model" , "data")] = data_ushaped
df_convergence.loc[:,  ("Ushaped model" , "converged parameters")] = parameter_convergence
df_convergence.loc[:,  ("Ushaped model" , "converged chains")] = chain_convergence
df_convergence.loc[:,  ("Ushaped model" , "converged runs")] = run_convergence
    
print(df_convergence)
latex = df_convergence.to_latex(index = False, column_format='lrrr', multicolumn_format='c')
latex = latex.replace('toprule', 'hline').replace('midrule','hline').replace('bottomrule','hline')
with open('7_Discussion/exp2/convergence_table_ushaped.tex', 'w') as f:
    f.write(latex)
print()


#%%
############# CODE FOR COMPUTING EFFECTIVE SAMPLE SIZE ###################

# for function in data_monotonic:
#     if function == 'fertility':
#         continue
#     else: 
#         parameters = np.load(path+f'/HS_model/results/{function}/params_and_rhat/10000_test_params.npy')
#         effective_sample_size = np.zeros((20,parameters.shape[1]))
#         for r in range(20):
#             chains = extract_chains(parameters[r,:,:], num_chains=4, num_samples=10000)
#             effective_sample_size[r] = compute_effective_sample_size(chains)
#     print(effective_sample_size)        
#     np.save(f'7_Discussion/exp1/{function}_effective_sample_size.npy', effective_sample_size)
#     print()




################### CODE FOR TABLES WITH TRADITIONAL RHAT ######################
print('Traditional rhat')
# MONOTONIC
print('\tMonotonic')
columns = pd.MultiIndex.from_tuples([   ("Monotonic model" , "data"),
                                        ("Monotonic model" , "converged parameters"),
                                        ("Monotonic model" , "converged runs"),
                                       ])

df_convergence = pd.DataFrame(columns=columns)

parameter_convergence = []
chain_convergence = []
run_convergence = []

data_monotonic = ['flat', 'sinusoidal', 'step', 'linear', 'exp', 'logical', 'fertility']

for data in data_monotonic:
    if data != 'fertility':
        path = '6_Experiments/61_Monotonic_benchmark_functions'
        parameters = np.load(path+f'/HS_model/results/{data}/params_and_rhat/10000_test_params.npy')
        rhat = np.zeros((parameters.shape[0:2]))
        for r in range(20):
            rhat[r,:] = compute_rhat_across_chains(parameter_array=parameters[r,:,:], num_chains=4, num_samples=10000)

        parameter_convergence.append(f'{np.sum(rhat <= 1.01)}/{np.prod(rhat.shape)}')
        runs_converged = np.prod(rhat <= 1.01, axis=1)
        run_convergence.append(f'{np.sum(runs_converged)}/20')

        if data == 'step':
            print(rhat.shape)
            for i in range(len(rhat)):
                print(f'{i}: ', np.sum(rhat[i] <= 1.01, axis= -1))
    
    else:
        path = '5_Shape_constrained_modelling/53_Relaxing_convexity'
        rhat = np.load(path + f'/HS_model/results/params_and_rhat/10000_test_rhat.npy')
        parameter_convergence.append(f'{np.sum(rhat <= 1.01)}/{np.prod(rhat.shape)}')
        run_convergence.append('nan')

df_convergence.loc[:,  ("Monotonic model" , "data")] = data_monotonic
df_convergence.loc[:,  ("Monotonic model" , "converged parameters")] = parameter_convergence
df_convergence.loc[:,  ("Monotonic model" , "converged runs")] = run_convergence

print(df_convergence)
print()

# USHAPED
print('\tUshaped')
columns = pd.MultiIndex.from_tuples([   ("Ushaped model" , "data"),
                                        ("Ushaped model" , "converged parameters"),
                                        ("Ushaped model" , "converged runs"),
                                       ])

df_convergence = pd.DataFrame(columns=columns)

parameter_convergence = []
chain_convergence = []
run_convergence = []

data_ushaped = ['flat', 'skew', 'parabola', 'abs', 'sine', 'step', 'depression']
for data in data_ushaped:
    if data != 'depression':
        path = '6_Experiments/63_Ushaped_benchmark_functions'
        parameters = np.load(path+f'/HS_model/results/{data}/params_and_rhat/10000_test_params_array.npy')
        rhat = np.zeros((parameters.shape[0:2]))
        for r in range(20):
            rhat[r,:] = compute_rhat_across_chains(parameter_array=parameters[r,:,:], num_chains=4, num_samples=10000)
        parameter_convergence.append(f'{np.sum(rhat <= 1.01)}/{np.prod(rhat.shape)}')
        runs_converged = np.prod(rhat <= 1.01, axis=1)
        run_convergence.append(f'{np.sum(runs_converged)}/20')
        if data == 'skew':
            print(rhat.shape)
            for i in range(len(rhat)):
                print(f'{i}: ', np.sum(rhat[i] <= 1.01, axis= -1))

    else:
        # path = '5_Shape_constrained_modelling/53_Relaxing_convexity'
        # rhat = np.load(path + f'/HS_model/results/params_and_rhat/10000_test_rhat.npy')
        # parameter_convergence.append(f'{np.sum(rhat <= 1.01)}/{np.prod(rhat.shape)}')
        # chains_converged = np.prod(rhat <= 1.01, axis=-1)
        # chain_convergence.append(f'{np.sum(chains_converged)}/{num_chains}')
        parameter_convergence.append('na')
        chain_convergence.append('na')
        run_convergence.append('na')


df_convergence.loc[:,  ("Ushaped model" , "data")] = data_ushaped
df_convergence.loc[:,  ("Ushaped model" , "converged parameters")] = parameter_convergence
df_convergence.loc[:,  ("Ushaped model" , "converged runs")] = run_convergence

print(df_convergence) 




######################## CODE FOR INCLUDING EFFECTIVE NUMBER OF SAMPLES ##############################
# columns = pd.MultiIndex.from_tuples([   ("Monotonic model" , "data"),
#                                         ("Monotonic model" , "converged parameters"),
#                                         ("Monotonic model" , "Parameters with $N_{eff} \geq 100$"),   
#                                         ("Monotonic model" , "converged chains"),                                
#                                         ("Monotonic model" , "converged runs"),
#                                         ("Monotonic model" , "Runs with $N_{eff} \geq 100$")
#                                        ])

# df_convergence = pd.DataFrame(columns=columns)

# parameter_convergence = []
# parameter_effective_samples = []
# chain_convergence = []
# run_convergence = []
# run_effective_samples = []

# data_monotonic = ['flat', 'sinusoidal', 'step', 'linear', 'exp', 'logical', 'fertility']
# for data in data_monotonic:
#     if data != 'fertility':
#         path = '6_Experiments/61_Monotonic_benchmark_functions'
#         # Asses rhat
#         rhat = np.load(path+f'/HS_model/results/{data}/params_and_rhat/10000_test_rhat.npy')
#         num_params = np.prod(rhat.shape)
#         parameter_convergence.append(f'{np.sum(rhat <= 1.01)}/{num_params}')
#         chains_converged = np.prod(rhat <= 1.01, axis=-1)
#         chain_convergence.append(f'{np.sum(chains_converged)}/{num_chains*20}')
#         runs_converged = np.prod(chains_converged, axis=-1)
#         run_convergence.append(f'{np.sum(runs_converged)}/20')

#         # Asses effective sample size
#         effective_sample_size = np.load(f'7_Discussion/exp1/{data}_effective_sample_size.npy')
#         param_eff = np.sum(effective_sample_size >= 100)
#         run_eff = np.sum(np.prod(effective_sample_size >= 100, axis=-1))
        
#         parameter_effective_samples.append(f'{param_eff}/{num_params}')
#         run_effective_samples.append(f'{run_eff}/20')

#     else:
#         parameter_effective_samples.append(f'na')
#         run_effective_samples.append(f'na')


# df_convergence.loc[:,  ("Monotonic model" , "data")] = data_monotonic
# df_convergence.loc[:,  ("Monotonic model" , "converged parameters")] = parameter_convergence
# df_convergence.loc[:,  ("Monotonic model" , "converged chains")] = chain_convergence
# df_convergence.loc[:,  ("Monotonic model" , "converged runs")] = run_convergence
# df_convergence.loc[:,  ("Monotonic model" , "Parameters with $N_{eff} \geq 100$")] = parameter_effective_samples
# df_convergence.loc[:,  ("Monotonic model" , "Runs with $N_{eff} \geq 100$")] = run_effective_samples

# print(df_convergence)

# print(df_convergence.to_latex(index = False, column_format='lrrr', multicolumn_format='c'))
