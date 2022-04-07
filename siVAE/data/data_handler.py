import os

import math
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pickle
import re

import copy

from siVAE.util import reduce_samples

def split_index(length, k_split, label = None, random_seed = 0):
    """ Given data and label, split into test/train then give result """
    list_index = []
    X0 = np.arange(length)
    if k_split < 1:
        if label is None:
            label = np.ones(length)
        _,counts  = np.unique(label, return_counts=True)
        if min(counts) > 1:
            test_data, train_data, _, _ = train_test_split(pd.DataFrame(X0), label,
                                                           test_size    = k_split,
                                                           stratify     = label,
                                                           random_state = random_seed)
        else:
            test_data, train_data, _, _ = train_test_split(pd.DataFrame(X0), label,
                                                           test_size    = k_split,
                                                           random_state = random_seed)

        split_values = (np.array(train_data.index), np.array(test_data.index))
        list_index.append(split_values)
    #
    elif k_split == 1:
        split_values = (X0, X0)
        list_index.append(split_values)
    #
    else:
        skf = StratifiedKFold(n_splits     = int(k_split),
                              random_state = random_seed,
                              shuffle      = True)
        for split_values in skf.split(X0, label):
            list_index.append(split_values)
    #
    return list_index


def split_data(data_list, split_idx, axis_list = None):
    """ Split list of data by train/test index for specificed axis"""

    train_index, test_index = split_idx
    dataset = []

    if axis_list is None:
        axis_list = [0] * len(data_list)

    for data in data_list:
        dataset.append(np.array(data[train_index]))
        dataset.append(np.array(data[test_index]))

    return dataset


def kfold_split(X,y, k_split, random_seed = 1, isPrecompMatrix = False):
    skf = StratifiedKFold(n_splits     = k_split,
                          random_state = random_seed,
                          shuffle      = True)
    X = np.array(X)
    y = np.array(y)
    if not isPrecompMatrix:
        datasets = [[X[idx_train],
                     X[idx_test],
                     y[idx_train],
                     y[idx_test]] for idx_train, idx_test in skf.split(X, y)]
    else:
        datasets = [[X[idx_train][:,idx_train],
                     X[idx_test][:,idx_train],
                     y[idx_train],
                     y[idx_test]] for idx_train, idx_test in skf.split(X, y)]
    return datasets


def convert2raw(adata):
    """ Convert the X in AnnData to raw expression from raw copy"""
    ##
    adata_raw = adata.raw.to_adata()
    adata_raw.var['idx'] = np.arange(adata_raw.var.shape[0])
    idx_keep = adata_raw.var.loc[adata.var_names].idx
    adata.X = adata_raw.X[:,idx_keep]

    if adata.isview:
        sc._utils.view_to_actual(adata)

    ## Convert sparse matrix to matrix
    try:
        adata.X = np.array(adata.X.toarray())
    except:
        pass

    return adata


def reduce_samples_adata(adata, mode, reduce_subset=None, **kwargs):
    """"""
    #
    adata_f = adata.copy()
    if reduce_subset is not None:
        adata_f = adata_f[np.isin(adata_f.obs['Labels'],reduce_subset)]
    del adata_f.raw
    #
    if mode == 'PCA':
        X = adata_f.X
        ## Convert sparse matrix to matrix
        try:
            X = np.array(X.toarray())
        except:
            pass
    elif mode == 'sample':
        X = adata_f
    else:
        raise Exception('Input valide mode for sample reducing')
    #
    X_red = reduce_samples(X, mode = mode,
                          label_in = adata_f.obs['Labels'], **kwargs)
    #
    if mode == 'PCA':
        adata_f   = adata_f[:kwargs['num_reduced']]
        adata_f.X = X_red
        adata_f.obs = pd.DataFrame({'Labels': ['PC-{}'.format(ii+1) for ii in range(adata_f.shape[0])]})
    elif mode == 'sample':
        adata_f = X_red
    #
    ## Transpose to n_gene x n_cell_reduced matrix
    adata_f = adata_f.copy().transpose()
    adata_f.obs['Labels'] = None
    return adata_f


def adata2datah(adata, mode='sample', k_split=None, random_seed=0,
                num_reduced=10000, reduce_mode='sample', reduce_subset=None):
    """ Convert AnnData to data handler"""
    ## Data handler

    if mode == 'sample':

        if adata.raw is not None:
            ## Load adata raw
            adata_raw = adata.raw.to_adata()
            adata_raw.var['idx'] = np.arange(adata_raw.var.shape[0])
            idx_keep = adata_raw.var.loc[adata.var_names].idx
            adata.raw = adata_raw[:,idx_keep]

        datah = data_handler(X=adata,y=adata)

    elif mode == 'feature':

        ## reduce samples
        adata_f = reduce_samples_adata(adata, mode=reduce_mode,
                                       num_reduced=num_reduced,
                                       reduce_subset=reduce_subset)

        ## Set raw object for adata_f
        if adata.raw is not None:
            ## Load adata raw and subset features
            adata_raw = adata.raw.to_adata()
            adata_raw.var['idx'] = np.arange(adata_raw.var.shape[0])
            idx_keep = adata_raw.var.loc[adata.var_names].idx
            adata_raw   = adata_raw[:,idx_keep]
            ## reduce samples
            adata_f_raw = reduce_samples_adata(adata_raw, mode=reduce_mode,
                                                num_reduced=num_reduced,
                                                reduce_subset=reduce_subset)
            adata_f.raw = adata_f_raw

        datah = data_handler(X=adata_f)

    ## k-fold split

    if k_split is not None:
        datah.create_split_index_list(k_split=k_split,random_seed=random_seed)

    return datah

class data_handler(object):
    """
    Handles input data.
    """
    def __init__(self, h_dims = None, X = None, y = None, label = None, batch = None):
        self.X = X
        self.y = y
        self.iterator = None
        self.batch=batch

    def create_split_index_list(self, k_split, random_seed = 0):
        """ """
        self.split_index_list = split_index(length      = len(self.X),
                                            k_split     = k_split,
                                            label       = self.X.obs['Labels'],
                                            random_seed = random_seed)

    def update_split_index_list(self, split_index_list):
        self.split_index_list = split_index_list

    def copy(self):
        """ Create a copy """
        return(copy.deepcopy(self))

    def create_dataset(self, kfold_idx=None):
        """
        split_idx: index of the training and test set in X and y data
        """
        if kfold_idx is None:
            split_idx = self.split_idx
        else:
            split_idx = self.get_kfold_index(kfold_idx)
            self.split_idx= split_idx

        data_list = [self.X.X,
                     self.y.X]

        if self.batch is not None:
            data_list.append(self.batch)

        self.dataset = split_data(data_list, split_idx)

    def get_kfold_index(self, kfold_idx):
        return self.split_index_list[kfold_idx]

    def convert2raw(self):
        """ Convert the adata X to raw """
        if self.X.raw is not None:
            adata = self.X.raw.to_adata()
            if adata.isview:
                sc._utils.view_to_actual(adata)

            ## Convert sparse matrix to matrix
            try:
                adata.X = np.array(adata.X.toarray())
            except:
                pass

            self.X.X = adata.X

        if self.y is not None:
            if self.y.raw is not None:
                adata = self.y.raw.to_adata()
                if adata.isview:
                    sc._utils.view_to_actual(adata)

                ## Convert sparse matrix to matrix
                try:
                    adata.X = np.array(adata.X.toarray())
                except:
                    pass

                self.y.X = adata.X

def data2handler(adata):
    """
    AnnData (n_cell x n_gene)
    obs and var should both include column ['Labels']
    """
    adata_f = adata.copy().transpose()
    adata_f.obs['Labels'] = None

    ## Set up data_handlers ====================================================

    ## Data handler per sample
    datah_sample = data_handler(X=adata,y=adata)
    datah_sample.create_split_index_list(k_split=0.8,random_seed=0)
    datah_sample.create_dataset(kfold_idx=0)

    ## Data handler per feature
    datah_feature = data_handler(X=adata_f)

    #### Manually set up additional factors
    labels_feature = np.array(datah_sample.X.var_names)
    label_feature_group = np.repeat('NA',datah_sample.X.shape[1])
    if 'Labels' not in datah_sample.X.obs.columns:
        datah_sample.X.obs['Labels'] = 'NA'
    labels = np.array(datah_sample.X.obs['Labels'])

    ## Set palette for all labels/label_feature_group
    keys = np.unique(labels)
    palette = None

    hue_order = np.array(keys)

    ## Set up dictionary of arguments for plotting
    plot_args_sample  = {'labels': labels,
                         'alpha' : 0.1,
                         'names' : None,
                         'hide_legend': False
                         }

    plot_args_feature = {'labels': label_feature_group,
                         'alpha' : 0.01,
                         'names' : labels_feature,
                         'hide_legend': False
                         }

    plot_args = {'feature'  : plot_args_feature,
                 'sample'   : plot_args_sample,
                 'hue_order': hue_order,
                 'palette'  : palette
                 }

    return(datah_sample, datah_feature, plot_args)
