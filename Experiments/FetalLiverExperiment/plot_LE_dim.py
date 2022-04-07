
"""
2021/11/30

Plot comparisons of LE_dim
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


logdir = 'out/LE_dim'

datadir = 'out/data'


methods = [dir for dir in os.listdir(logdir) \
           if os.path.isdir(os.path.join(logdir,dir)) \
           and 'pca' not in dir and 'umap' not in dir]

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
            try:
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
                    df_recon['LE dim'] = int(LE_dim)
                    #
                    df_clf = pd.read_csv(os.path.join(logdir_kfold,'clf_accuracy.csv'))
                    df_clf.columns = ['VAE',method]
                    df_clf = df_clf.melt(value_name = 'Accuracy',
                                         value_vars=['VAE',method],
                                         var_name='Method')
                    df_clf['kfold'] = kfold
                    df_clf['LE dim'] = int(LE_dim)
                else:
                    pass
                #
                df_list.append(df_recon)
                df_clf_list.append(df_clf)
            except:
                pass

#### PCA/UMAP

# Load data
datadir = os.path.join('out/LE_dim/data_dict.pickle')
data_dict = util.load_pickle(datadir)
datah_sample  = data_dict['sample']

## Iterate over k-fold
LE_dims = np.unique(pd.concat(df_list)['LE dim'])
k_split = len(np.unique(pd.concat(df_list)['kfold']))

for method in ['umap','pca']:
    #
    logdir_method = os.path.join(logdir,method)
    os.makedirs(logdir_method,exist_ok=True)
    #
    results_clf_dict = {}
    for LE_dim in LE_dims:
        results_clf = []
        for k_fold in range(int(max(k_split,1))):
            #
            logdir_kfold = os.path.join(logdir_method,'kfold-{}'.format(k_fold))
            #
            datah_sample.create_dataset(kfold_idx=k_fold)
            y = datah_sample.X.obs['Cell Type'].to_numpy()
            y_train = y[datah_sample.get_kfold_index(k_fold)[0]]
            y_test  = y[datah_sample.get_kfold_index(k_fold)[1]]
            X_train = datah_sample.dataset[0]
            X_test  = datah_sample.dataset[1]
            #
            if method == 'pca':
                trans = PCA(LE_dim)
            elif method == 'umap':
                trans = umap.UMAP(n_components = LE_dim,
                                  random_state = 0)
            #
            trans.fit(X_train)
            #
            X_dict = {'train': X_train,
                      'test' : X_test}
            #
            if method == 'pca':
                pca = trans
                losses = []
                for data_type, X_in in X_dict.items():
                    X_emb = pca.transform(X_in)
                    X_recon = np.dot(X_emb, pca.components_) + pca.mean_
                    recon_error = np.square(X_recon - X_in).sum(1).mean()
                    losses.append(recon_error)
                #
                df_loss = pd.DataFrame([losses], columns = X_dict.keys())
                df_loss['method'] = 'pca'
                df_loss['kfold']  = 'kfold'+str(k_fold)
                df_loss['LE dim'] = LE_dim
                df_list.append(df_loss)
            #
            ## Run 5-fold classifier experiment on test data
            X_latent = trans.transform(X_test)
            skf = StratifiedKFold(n_splits=5)
            skf.get_n_splits(X_latent, y_test)
            #
            for train_index, test_index in skf.split(X_latent, y_test):
                X_train_ = X_latent[train_index]
                X_test_  = X_latent[test_index]
                y_train_ = y_test[train_index]
                y_test_  = y_test[test_index]
                n_neighbors = len(np.unique(y)) * 2
                result = classifier.run_classifier(X_train = X_train_, X_test = X_test_,
                                          y_train = y_train_, y_test = y_test_,
                                          classifier = "KNeighborsClassifier", max_dim = 1e6,
                                          args = {'n_neighbors':n_neighbors})[0]
                # result = classifier.run_classifier(X_train = X_train_, X_test = X_test_,
                #                           y_train = y_train_, y_test = y_test_,
                #                           classifier = "MLPClassifier", max_dim = 1e6,
                #                           )[0]
                results_clf.append(result)
        #
        results_clf_dict[LE_dim] = np.array(results_clf)
    #
    df_clf_accuracy = pd.DataFrame(results_clf_dict)
    df_clf_accuracy.to_csv(os.path.join(logdir_method,'clf_accuracy.csv'),index=False)
    df_clf_accuracy = df_clf_accuracy.melt(var_name='LE dim',value_name = 'Accuracy')
    df_clf_accuracy['Method'] = method
    df_clf_list.append(df_clf_accuracy)


#### Plots
# Set palette
methods_plot = ['siVAE','siVAE-0','VAE','pca','umap']
palette = {method:color for method,color in zip(methods_plot,sns.color_palette())}

## Classification accuracy
df_clf = pd.concat(df_clf_list)
df_clf['LE dim'] = df_clf['LE dim'].astype('int')
df_clf['Method'] = pd.Categorical(df_clf['Method'],
                                  categories = methods_plot,
                                  ordered=True)
df_clf = df_clf.sort_values('Method')

sns.barplot(data = df_clf,
            x = 'LE dim',
            y = 'Accuracy',
            hue = 'Method',
            )
plt.savefig(os.path.join(logdir,f'clf_accuracy.svg'))
plt.close()

## Recon losses
df_losses = pd.concat(df_list)
df_losses['LE dim'] = df_losses['LE dim'].astype('int')
df_losses.to_csv(os.path.join(logdir,'recon_losses.csv'))
df_losses['order'] = pd.Categorical(df_losses['method'],
                                     categories = methods_plot,
                                     ordered=True)
df_losses = df_losses.sort_values('order')

for data_type in ['train','test']:
    sns.barplot(data = df_losses,
                x = 'LE dim',
                y = data_type,
                hue = 'method')
    plt.savefig(os.path.join(logdir,f'recon_loss-{data_type}.svg'))
    plt.close()

################################################################################
