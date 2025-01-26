#%%
import numpy as np
import pandas as pd
import pickle

R = 20
path = '6_Experiments/61_Monotonic_benchmark_functions/'

models = ['baseline_GP', 'HS_model', 'VP_model', 'SDE_model']
model_names = [model.replace('_', ' ') for model in models]

benchmark_functions = ['flat', 'sinusoidal', 'step', 'linear', 'exp', 'logical']

df_models = pd.DataFrame(index=benchmark_functions)
df_RMSE = pd.DataFrame(index=benchmark_functions)
df_lpd  = pd.DataFrame(index=benchmark_functions)

for m, model in enumerate(models):
    model_parameters = []
    RMSE_means = []
    RMSE_stds = []
    lpd_means = []
    lpd_stds = []
    
    for benchmark_function in benchmark_functions:
        resultpath = path + f'{model}/results/{benchmark_function}/'
        result = np.load(resultpath + '10000_test_results.npy')
        
        RMSE_means.append(result[0])
        RMSE_stds.append('$\pm$ ' + str((result[1]/np.sqrt(R)).round(3)))
        lpd_means.append(result[2])
        lpd_stds.append('$\pm$ ' + str((result[3]/np.sqrt(R)).round(3)))

        # Get model parameters:
        if model != 'baseline_GP':
            with open(resultpath+'model_selection_results.pkl', 'rb') as f:
                    model_selection_results = pickle.load(f)

            model_params = ''
            for key,val in model_selection_results.items():
                model_params += key.replace('_', ' ') + ': ' + str(val) + ', '
            model_parameters.append(model_params[:-2])

    if model != 'baseline_GP':
        df_models[model_names[m]] = model_parameters
    df_RMSE[model_names[m]] = RMSE_means
    df_RMSE[m] = RMSE_stds

    df_lpd[model_names[m]] = lpd_means
    df_lpd[m] = lpd_stds

# df.loc['mean rank'] = df[model_names].rank(axis=1).mean()

def style_df(df, model_names, max=False, caption='Caption'):
    latex = '\\begin{table}[] \n \\centering\n'
    if max:
        rank = df[model_names].rank(axis=1, ascending=False).mean()
        styled_df = df.style.highlight_max(subset = model_names, 
                            props="textbf:--rwrap;", axis = 1)        
    else:
        rank = df[model_names].rank(axis=1).mean()
        styled_df = df.style.highlight_min(subset = model_names, 
                            props="textbf:--rwrap;", axis = 1)
    styled_df = styled_df.format("{:.3f}", subset=model_names)

    latex += styled_df.to_latex(hrules=True)
    for m, model in enumerate(model_names):
        latex = latex.replace(model + ' & ' + str(m), ' \multicolumn{2}{c}{' + model + '} ')
    latex = latex.replace('nan', '')
    latex = latex.replace('\\end{tabular}\n', '')
    latex = latex.replace('toprule', 'hline').replace('midrule', 'hline').replace('bottomrule', 'hline')

    latex += '\\textbf{mean rank}'

    for r in rank:
        latex += ' & \\multicolumn{2}{c}{\\textbf{' + str(round(r, 2)) + '}}'
    latex += '\\\\\n\\hline\n\\end{tabular}\n\\caption{'+caption+'}\n\\label{tab:'+caption+'}\n\\end{table}\n'
    return latex

latex = '\\begin{table}[] \n \\centering\n'
latex += df_models.to_latex()
latex = latex.replace('toprule', 'hline').replace('midrule', 'hline').replace('bottomrule', 'hline')
latex += '\\caption{model selection}\n\\label{tab:my_label}\n\\end{table}\n'
with open(path + 'model_selction_table.tex', 'w') as f:
    f.write(latex)

print(latex)
print()

latex = style_df(df_RMSE, model_names, caption='rmse')
with open(path + 'RMSE_table.tex', 'w') as f:
    f.write(latex)

print(latex)
print()


latex = style_df(df_lpd, model_names, max=True, caption='lpd')
with open(path + 'LPD_table.tex', 'w') as f:
    f.write(latex)

print(latex)
print()
