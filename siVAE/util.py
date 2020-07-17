import os
import time
import logging

import multiprocessing

import pickle
import gzip
import bz2

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split

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


def reduce_dimensions(X, reduced_dimension = 2, method = 'PCA', tsne_min = 50, match_dims = True):
    """
    Reduce dimensioanlity of X using dimensionality reduction method.
    """
    if X.shape[-1] > reduced_dimension:

        if method == "PCA":
            pca = PCA(n_components = reduced_dimension)
            pca.fit(X)
            X_pca = pca.transform(X)
            X_reduced = X_pca[:,:reduced_dimension]
            dim_name = 'pc'

        elif method == "tSNE":
            if X.shape[-1] > tsne_min:
                X,_ = reduce_dimensions(X, reduced_dimension = 50, method = 'PCA')
            tsne = TSNE(n_components = reduced_dimension, init = 'pca')
            X_tsne = tsne.fit_transform(X)
            X_reduced = X_tsne[:,:2]
            dim_name = 't-SNE'
    else:

        if X.shape[-1] < reduced_dimension and match_dims:
            dim_diff = reduced_dimension - X.shape[-1]
            X_reduced = np.concatenate([X, np.zeros([len(X),dim_diff])], axis = 1)
        else:
            X_reduced = X

        dim_name = "dim"

    dim_labels = ["{}_{}".format(dim_name, ii) for ii in range(reduced_dimension)]
    return X_reduced, dim_labels


def save_df_as_npz(filename, df):
    np.savez(filename, matrix = df.values, columns = df.columns, index = df.index, allow_pickle = True)


def load_df_from_npz(filename):
    uploaded = np.load(filename, allow_pickle = True)
    df = pd.DataFrame(uploaded['matrix'], columns = uploaded['columns'], index = uploaded['index'])
    return df


def reduce_samples(X, label_in, num_reduced = 10000, type='all'):
    """
    type = ['sample', 'PCA', 'all']
    """

    if type == 'sample':
        test_size= num_reduced/len(X)
        if test_size < 1:
            _,X_new,_,_ = train_test_split(X,label_in,test_size=test_size,stratify=label_in)

    elif type == 'PCA':
        X_t         = X.transpose()
        num_reduced = min(num_reduced, X_t.shape[0])
        X_new , _   = reduce_dimensions(X_t, reduced_dimension = num_reduced, method = 'PCA')
        X_new       = X_new.transpose()

    elif type == 'all':
        X_new = X

    else:
        raise Exception('Input valid VAE_feature_input ')

    return X_new
