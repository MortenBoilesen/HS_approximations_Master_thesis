#%%
import numpy as np
import pandas as pd
import pickle

R = 6
path = '6_Experiments/64_Ushaped_data/'
path = ''

models = ['baseline_GP', 'HS_model', 'VP_model']
model_names = ['baseline GP', 'HS model', 'VP model']

fractions=[0.5, 0.7272727272727273, 0.8888888888888888, 1.0, 2.0, 4.0]
num_train= [int(42*fraction/8) for fraction in fractions]

df_RMSE = pd.DataFrame(index=num_train)
df_lpd  = pd.DataFrame(index=num_train)

for m, model in enumerate(models):
    model_parameters = []
    RMSE_means = []
    RMSE_stds = []
    lpd_means = []
    lpd_stds = []
    
    for fraction in fractions:
        rmses = np.zeros(R)
        elpds = np.zeros(R)
        
        for r in range(R):
            
            try:
                resultpath = path + f'Runs/Run_{r+1}/{model}/{fraction}/'
                result = np.load(resultpath + 'test_results.npy')
            except:
                resultpath = path + f'Runs/Run_{r+1}/{model}/{int(fraction)}/'
                result = np.load(resultpath + 'test_results.npy')
        
            rmses[r] = result[0]
            elpds[r] = result[1]


        RMSE_means.append(rmses.mean())
        RMSE_stds.append('$\pm$ ' + str((rmses.std()/np.sqrt(R)).round(3)))
        lpd_means.append(elpds.mean())
        lpd_stds.append('$\pm$ ' + str((elpds.std()/np.sqrt(R)).round(3)))

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


latex = style_df(df_RMSE, model_names, caption='rmse')
print(latex)
with open(path + 'RMSE_table.tex', 'w') as f:
    f.write(latex)

print()
latex = style_df(df_lpd, model_names, max=True, caption='lpd')
with open(path + 'LPD_table.tex', 'w') as f:
    f.write(latex)
print(latex)