import os
import time
import logging

import multiprocessing

import pickle
import gzip
import bz2

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split

## Plots
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

def run_processor(fun, *args, **kwargs):
    """Run a function in processor using multiprocessing"""
    # Input: function to be run, args to the
    # Return: Output of the input fun
    def run_fun(fun, return_dict, args, kwargs):
        return_dict[1] = fun(*args, **kwargs)
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    b = multiprocessing.Process(target=run_fun, args=(fun, return_dict, args, kwargs))
    b.start()
    b.join()
    return return_dict[1]


def mat2onehot(mat, category = None):
    """ Turns matrix into one hot format"""
    mat = np.array(mat)
    if category == None:
        category = np.unique(mat)
    mat_onehot = [np.isin(category, array) for array in mat]
    mat_onehot = np.array(mat_onehot).astype('int')
    return mat_onehot


def multiple_hot_df(item_lists):
    """ Merge data frames """
    fullset = []
    for item_list in item_lists:
        fullset += np.array(item_list)[np.invert(np.isin(item_list, fullset))].tolist()
    mat_bin = np.array([np.isin(fullset,item_list) for item_list in item_lists]).astype('int')
    mat_bin = pd.DataFrame(mat_bin.transpose())
    mat_bin.index = fullset
    return mat_bin


def makedirs(logdir):
    Path(logdir).mkdir(parents = True, exist_ok = True)


def GetSpacedElements(array, numElems = 4):
    out = array[np.round(np.linspace(0, len(array)-1, numElems)).astype(int)]
    return out


def save_pickle(obj, filename, zip_method = None, protocol=-1, correct_suffix = True):
    """"""
    if correct_suffix:
        filename = filename.rsplit(".",1)[0]
        if zip_method is None:
            filename += ".pickle"
        elif zip_method is 'gzip':
            filename += ".pgz"
        elif zip_method is 'bz2':
            filename += ".pbz2"
        else:
            raise Exception('Specify a correct zip_method: {}'.format(zip_method))

    if zip_method is None:
        with open(filename, 'wb') as f:
            pickle.dump(obj, f, protocol)
    elif zip_method == 'gzip':
        with gzip.open(filename, 'w') as f:
            pickle.dump(obj, f)
    elif zip_method == 'bz2':
         with bz2.BZ2File(filename, 'w') as f:
            pickle.dump(obj, f)
    else:
        raise Exception('Specify a correct zip_method: {}'.format(zip_method))


def load_pickle(filename, zip_method=None, correct_suffix = True):

    if correct_suffix:
        filename = filename.rsplit(".",1)[0]
        if zip_method is None:
            filename += ".pickle"
        elif zip_method is 'gzip':
            filename += ".pgz"
        elif zip_method is 'bz2':
            filename += ".pbz2"
        else:
            raise Exception('Specify a correct zip_method: {}'.format(zip_method))

    if zip_method is None:
        with open(filename, 'rb') as f:
            loaded = pickle.load(f)
    elif zip_method == 'gzip':
        with gzip.open(filename, 'r') as f:
            loaded = pickle.load(f)
    elif zip_method == 'bz2':
         with bz2.BZ2File(filename, 'r') as f:
            loaded = pickle.load(f)
    else:
        raise Exception('Specify a correct zip_method: {}'.format(zip_method))

    return loaded


def reduce_dimensions(X, scale=False, reduced_dimension = 2, method = 'PCA', tsne_min = 50, match_dims = True,**kwargs):
    """
    Reduce dimensioanlity of X using dimensionality reduction method.
    """
    if X.shape[-1] > reduced_dimension:
        #
        if method == "PCA":
            pca = PCA(n_components = reduced_dimension,**kwargs)
            pca.fit(X)
            X_pca = pca.transform(X)
            X_reduced = X_pca[:,:reduced_dimension]
            dim_name = 'PC'
            #
        elif method == "tSNE":
            if X.shape[-1] > tsne_min:
                X,_ = reduce_dimensions(X, reduced_dimension = tsne_min, method = 'PCA')
            tsne = TSNE(n_components = reduced_dimension, init = 'pca', **kwargs)
            X_tsne = tsne.fit_transform(X)
            X_reduced = X_tsne[:,:2]
            dim_name = 't-SNE'
            #
        elif method == 'UMAP':
            if X.shape[-1] > tsne_min:
                X,_ = reduce_dimensions(X, reduced_dimension = tsne_min, method = 'PCA')
            X_reduced = umap.UMAP(**kwargs).fit_transform(X)
            X_reduced = X_reduced[:,:2]
            dim_name = 'UMAP'
            #
        else:
            raise Exception('{} is not a valid DR method (PCA,tSNE,UMAP)'.format(method))
            #
    else:
        #
        if X.shape[-1] < reduced_dimension and match_dims:
            dim_diff = reduced_dimension - X.shape[-1]
            X_reduced = np.concatenate([X, np.zeros([len(X),dim_diff])], axis = 1)
        else:
            X_reduced = X
            #
        dim_name = "Dim"
        #
    dim_labels = ["{} {}".format(dim_name, ii+1) for ii in range(reduced_dimension)]
    return X_reduced, dim_labels


def save_df_as_npz(filename, df):
    np.savez(filename, matrix = df.values, columns = df.columns, index = df.index, allow_pickle = True)


def load_df_from_npz(filename):
    uploaded = np.load(filename, allow_pickle = True)
    df = pd.DataFrame(uploaded['matrix'], columns = uploaded['columns'], index = uploaded['index'])
    return df


def reduce_samples(X, label_in, num_reduced = 10000, mode='sample'):
    """
    type = ['sample', 'PCA']
    """
    #
    if X.shape[0] > num_reduced:
        #
        if mode == 'sample':
            test_size= num_reduced/len(X)
            if test_size < 1:
                _,X_new,_,_ = train_test_split(X,label_in,test_size=test_size,stratify=label_in)
                #
        elif mode == 'PCA':
            X_t         = X.transpose()
            num_reduced = min(num_reduced, X_t.shape[0])
            X_new , _   = reduce_dimensions(X_t, reduced_dimension = num_reduced, method = 'PCA')
            X_new       = X_new.transpose()
            #
        else:
            raise Exception('Input valid reduce mode')
            #
    else:
        #
        X_new = X
        #
    return X_new


def remove_spines(ax,yaxis=True,xaxis=True,num_ticks=3,show_legend=True):
    if yaxis:
        ax.yaxis.set_major_locator(plt.MaxNLocator(num_ticks))
    if xaxis:
        ax.xaxis.set_major_locator(plt.MaxNLocator(num_ticks))
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ##
    if show_legend:
        # lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        legend = plt.legend(edgecolor='black')
        legend.get_frame().set_alpha(1)
    else:
        if ax.legend_ is not None:
            ax.legend_.remove()
