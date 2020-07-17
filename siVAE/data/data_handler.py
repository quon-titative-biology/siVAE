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

def split_index(length, k_split, label = None, random_seed = 0):
    """ Given data and label, split into test/train then give result """
    list_index = []
    X0 = np.arange(length)
    if k_split < 1:
        if label is None:
            label = np.ones(length)
        test_data, train_data, _, _ = train_test_split(pd.DataFrame(X0), label,
                                                       test_size    = k_split,
                                                       stratify     = label,
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
    #
    if axis_list is None:
        axis_list = [0] * len(data_list)
    #
    for data in data_list:
        dataset.append(data[train_index])
        dataset.append(data[test_index])
    #
    return dataset


def kfold_split(X,y, k_split, random_seed = 1, isPrecompMatrix = False):
    skf = StratifiedKFold(n_splits     = k_split,
                          random_state = random_seed,
                          shuffle      = True)
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


class data_handler(object):
    """
    Main Data (X, y)
    """
    def __init__(self, h_dims = None, X = None, y = None, label = None):
        self.X = X
        self.y = y
        self.iterator = None
    #
    def create_split_index_list(self, k_split, random_seed = 0):
        """ """
        self.split_index_list = split_index(length      = len(self.X),
                                            k_split     = k_split,
                                            label       = self.X.obs['Labels'],
                                            random_seed = random_seed)
    #
    def copy(self):
        return(copy.deepcopy(self))
    #
    def create_dataset(self, kfold_idx=None):
        """
        split_idx: index of the training and test set in X and y data
        """
        if kfold_idx is None:
            split_idx = self.split_idx
        else:
            split_idx = self.get_kfold_index(kfold_idx)
            self.split_idx = split_idx
        data_list = [self.X.X,
                     self.y.X]
        self.dataset = split_data(data_list, split_idx)
    #
    def get_kfold_index(self, kfold_idx):
        return self.split_index_list[kfold_idx]
