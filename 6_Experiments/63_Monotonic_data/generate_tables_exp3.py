#%%
import numpy as np
import pandas as pd
import pickle
R = 20
# path = '5_Shape_constrained_modelling/53_Relaxing_convexity/'
path = ''

models = ['baseline_GP', 'HS_model', 'VP_model', 'SDE_model']
model_names = [model.replace('_', ' ') for model in models]
model_parameters = []

df_RMSE = pd.DataFrame(index=['mean'])
df_lpd  = pd.DataFrame(index=['mean'])

for m, model in enumerate(models):
    resultpath = path + f'{model}/results/'
    if model != 'SDE_model':
        result = np.load(resultpath + '10000_test_results.npy')
    else:
        result = np.load(resultpath + 'test_results.npy')

    if model == 'SDE_model':
        df_RMSE[model_names[m]] = result[1]
        df_lpd[model_names[m]] = result[0]
    else:
        df_RMSE[model_names[m]] = result[0]
        df_lpd[model_names[m]] = result[1]

    if model == 'baseline_GP':
        model_parameters.append('~')
    else:
        with open(resultpath+'model_selection_results.pkl', 'rb') as f:
                model_selection_results = pickle.load(f)

        model_params = ''
        for key,val in model_selection_results.items():
            model_params += key.replace('_', ' ') + ': ' + str(val) + ', '
        model_parameters.append(model_params[:-2])


def style_df(df, model_names, max = False, caption='Caption'):
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
        # latex = latex.replace(model + ' & ' + str(m), ' \multicolumn{2}{c}{' + model + '} ')
        latex = latex.replace(model_names[m], '\\begin{tabular}[c]{@{}l@{}}' + model + '\\\\' + model_parameters[m] + '\end{tabular}')


    latex = latex.replace('nan', '')
    latex = latex.replace('\\end{tabular}\n', '')
    latex = latex.replace('toprule', 'hline').replace('midrule', 'hline').replace('bottomrule', 'hline')
    latex = latex.replace('_', '\_')

    latex += '\\textbf{rank}'

    
    for r in rank:
        latex += ' & \\textbf{' + str(round(r, 2)) + '}'
    latex += '\\\\\n\\hline\n\\end{tabular}\n\\caption{'+caption+'}\n\\label{tab:'+caption+'}\n\\end{table}\n'
    return latex


latex = style_df(df_RMSE, model_names, caption='rmse')
with open(path + 'RMSE_table.tex', 'w') as f:
    f.write(latex)

print(latex)

latex = style_df(df_lpd, model_names, max=True, caption='lpd')
with open(path + 'LPD_table.tex', 'w') as f:
    f.write(latex)

print(latex)

# %%
