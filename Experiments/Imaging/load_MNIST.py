import os

import copy

import tensorflow as tf

import numpy as np
import pandas as pd
import math
import random

import matplotlib.colors as colors

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from siVAE.data import data_handler as dh
from siVAE.util import reduce_dimensions
from siVAE.util import save_df_as_npz
from siVAE.util import load_df_from_npz
from siVAE.util import reduce_samples
from siVAE.model.output.analysis import getPCALoadings

import scanpy as sc
import anndata

def load_adata(subsets_labels = None, max_size = 100000, scale=True, dataset = 'MNIST'):

    """ Create annotated data """

    random.seed(0)

    ## Load data from tensorflow
    if dataset == 'MNIST':
        MNIST_datas = tf.keras.datasets.mnist.load_data()
        ImageDims = [28,28,1]
    elif dataset == 'FMNIST':
        MNIST_datas = tf.keras.datasets.fashion_mnist.load_data()
        ImageDims = [28,28,1]
    elif dataset == 'CIFAR10':
        MNIST_datas = tf.keras.datasets.cifar10.load_data()
        ImageDims = [32,32,3]

    test_set = MNIST_datas[1]
    train_set = MNIST_datas[0]

    ## Assign test set to X, and flatten
    use_test_set = False

    if use_test_set:
        X = test_set[0]
        labels = test_set[1]
    else:
        X = train_set[0]
        labels = train_set[1]

    ## Subset by label
    if subsets_labels is not None:
        idx_keep = np.isin(labels,subsets_labels)
        X = X[idx_keep]
        labels = labels[idx_keep]

    labels = labels.astype('str')
    labels = np.core.defchararray.add('digit-', labels)
    X = X/255
    X = X.reshape(len(X),-1)

    obs = pd.DataFrame({'Labels': labels.reshape(-1)})

    adata = anndata.AnnData(X=X,obs=obs)
    adata.var['Labels'] = None

    ## Subsample
    test_size = 1-max_size/len(adata)
    if test_size > 0:
        adata,_ = train_test_split(adata,test_size = test_size,stratify=adata.obs['Labels'])

    adata.raw = adata
    if scale:
        sc.pp.scale(adata)

    palette_manual = None

    return(adata,palette_manual)


def prepare_data(num_reduced,sample_size=100,**kwargs):

    random.seed(0)

    ImageDims = [28,28,1]

    adata, palette_manual = load_adata(**kwargs)


    ## Set up data_handlers ====================================================

    datah_sample  = dh.adata2datah(adata,mode='sample')
    datah_feature = dh.adata2datah(adata,mode='feature',num_reduced=num_reduced)

    #### Manually set up additional factors
    labels_feature = np.array(datah_sample.X.var_names)
    label_feature_group = np.repeat('NA',datah_sample.X.shape[1])
    labels = np.array(datah_sample.X.obs['Labels'])

    ## Set palette for all labels/label_feature_group
    keys = np.concatenate([np.unique(label_feature_group),np.unique(labels)])
    keys = np.unique(keys)
    cnames = list(colors.cnames.keys())
    random.seed(0)
    random.shuffle(cnames)

    palette = {key: cnames[ii] for ii,key in enumerate(keys)}

    hue_order = np.unique(adata.obs['Labels'])
    hue_order = np.array(hue_order)[np.isin(hue_order,keys)]
    hue_order = np.concatenate([hue_order,keys[np.invert(np.isin(keys,hue_order))]])

    baseline = np.min(datah_sample.X.X,0)
    kwargs_FI = {'baseline': baseline}

    ## Set up dictionary of arguments for plotting
    metadata      = np.array(datah_sample.X.obs)
    metadata_name = np.array(datah_sample.X.obs.columns)
    plot_args_sample = {'labels': labels,
                        'metadata': metadata,
                        'metadata_name': metadata_name,
                        'alpha' : 0.1,
                        'names' : None,
                        'hide_legend': False
                        }

    plot_args_feature = {'labels': label_feature_group,
                         'alpha' : 0.01,
                         'names' : labels_feature,
                         'hide_legend': False
                         }

    if palette_manual is not None:
        palette = palette_manual


    plot_args_dict_sf = {'feature'  : plot_args_feature,
                         'sample'   : plot_args_sample,
                         'hue_order': hue_order,
                         'palette'  : palette,
                         'ImageDims': ImageDims
                         }

    ## Set up Samples
    test_size = 1 - sample_size/len(adata)
    if test_size > 0:
        adata_sample,_ = train_test_split(adata,test_size = test_size,stratify=adata.obs['Labels'])

    # sample_idx = [52599, 10638, 26875, 27828, 48669, 18326, 22803, 10815, 12904, 47878]
    #
    # adata_sample = adata[sample_idx]

    sample_set  = {'labels'  : adata_sample.obs['Labels'],
                   'samples' : adata_sample.X}

    return(kwargs_FI, sample_set, datah_feature, datah_sample, plot_args_dict_sf)
