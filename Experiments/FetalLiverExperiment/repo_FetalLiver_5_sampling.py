#### Analysis on experiments

# Built-in
import os
import gc

# Libs
import numpy as np
import pandas as pd
from scipy import stats

# Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap

logdirbase='out/siVAE_reduce'

exps = os.listdir(logdirbase)
df_exp_meta = pd.DataFrame([exp.split("-",2) for exp in exps],
                           columns = ['Method', 'Reduced number', 'Cell type'])
df_exp_meta.index = exps
df_exp_meta['Reduced number'] = df_exp_meta['Reduced number'].astype('int')
df_exp_meta = df_exp_meta[df_exp_meta['Cell type'].isnull()]
df_exp_meta = df_exp_meta[df_exp_meta['Reduced number'] != 1]

df_plots = []
for exp in df_exp_meta.index:
    resultdir = os.path.join(logdirbase, exp, 'kfold-0')
    clf = pd.read_csv(os.path.join(resultdir,'clf_accuracy.csv'))
    losses = pd.read_csv(os.path.join(resultdir,'losses.csv'))
    df_new = pd.DataFrame({'Classification': clf['model'].values,
                           'Losses': losses.iloc[0,1]})
    df_new['Method'] = df_exp_meta.loc[exp].Method
    df_new['Reduced number'] = df_exp_meta.loc[exp]['Reduced number']
    df_new['Name'] =  exp
    df_plots.append(df_new)

df_plots = pd.concat(df_plots)
df_plots = df_plots.sort_values(['Method','Reduced number'])

for metric in ['Classification', 'Losses']:
    g = sns.barplot(data=df_plots, x='Name', y= metric, hue = 'Method', dodge=False)
    g.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(logdirbase,f'{metric}.svg'))
    plt.close()

g = sns.barplot(data=df_plots, x='Name', y= 'Classification', hue = 'Method', dodge=False)
g.tick_params(axis='x', rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(logdirbase,'classification.svg'))
plt.close()
