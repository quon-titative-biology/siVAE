
"""
2021/11/30

Plot comparisons of zv_recon_scale
"""

import os

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
import umap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold

from siVAE import util
from siVAE import classifier


logdir = 'out/zv_recon_scale'

datadir = 'out/data'


methods = [dir for dir in os.listdir(logdir) \
           if os.path.isdir(os.path.join(logdir,dir)) \
           and 'pca' not in dir and 'umap' not in dir]

varname = 'zv_recon_scale'

df_list = []
df_clf_list = []
for method in methods:
    #
    logdir_method = os.path.join(logdir,method)
    #
    # for LE_dim in os.listdir(logdir_method):
    for LE_dim in os.listdir(logdir_method):
        #
        logdir_model = os.path.join(logdir_method, LE_dim)
        #
        kfolds = [k for k in os.listdir(logdir_model) if 'kfold' in k]
        for kfold in kfolds:
            if 'siVAE' in method:
                # Load losses
                logdir_kfold = os.path.join(logdir_model, kfold)
                df_siVAE = pd.read_csv(os.path.join(logdir_kfold,'losses.csv'))
                df_VAE = pd.read_csv(os.path.join(logdir_kfold,'losses_sample.csv'))
                # Combine to df
                df_recon = pd.concat([df_siVAE[df_siVAE.name == 'recon_loss'],
                                      df_VAE[df_VAE.name == 'recon loss']]
                                    )
                df_recon['method'] = [method,'VAE']
                df_recon['kfold'] = kfold
                df_recon[varname] = LE_dim
                #
                df_clf = pd.read_csv(os.path.join(logdir_kfold,'clf_accuracy.csv'))
                df_clf.columns = ['VAE',method]
                df_clf = df_clf.melt(value_name = 'Accuracy',
                                     value_vars=['VAE',method],
                                     var_name='Method')
                df_clf['kfold'] = kfold
                df_clf[varname] = LE_dim
            else:
                pass
            #
            df_list.append(df_recon)
            df_clf_list.append(df_clf)

#### Plots
# Set palette
methods_plot = ['siVAE']
palette = {method:color for method,color in zip(methods_plot,sns.color_palette())}

## Classification accuracy
df_clf = pd.concat(df_clf_list)
df_clf = df_clf[df_clf.Method == 'siVAE']
df_clf[varname] = df_clf[varname].astype('float32')
df_clf['Method'] = pd.Categorical(df_clf['Method'],
                                  categories = methods_plot,
                                  ordered=True)
df_clf = df_clf.sort_values('Method')

sns.barplot(data = df_clf,
            x = varname,
            y = 'Accuracy',
            hue = 'Method',
            )
plt.savefig(os.path.join(logdir,f'clf_accuracy.svg'))
plt.close()

## Recon losses
df_losses = pd.concat(df_list)
df_losses = df_losses[df_losses.method == 'siVAE']
df_losses[varname] = df_losses[varname].astype('float32')
df_losses.to_csv(os.path.join(logdir,'recon_losses.csv'))
df_losses['order'] = pd.Categorical(df_losses['method'],
                                     categories = methods_plot,
                                     ordered=True)
df_losses = df_losses.sort_values('order')

for data_type in ['train','test']:
    sns.barplot(data = df_losses,
                x = varname,
                y = data_type,
                hue = 'method')
    plt.savefig(os.path.join(logdir,f'recon_loss-{data_type}.svg'))
    plt.close()

################################################################################
