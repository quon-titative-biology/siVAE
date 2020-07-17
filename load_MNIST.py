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

import scanpy as sc
import anndata

def load_adata(subsets_labels = None):

    """ Create annotated data """

    random.seed(0)

    ## Load MNIST data
    MNIST_datas = tf.keras.datasets.mnist.load_data()
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

    labels = labels.astype('str')
    labels = np.core.defchararray.add('digit-', labels)
    X = X.reshape(len(X),-1)
    ImageDims = [28,28,1]

    obs = {'Labels': labels}

    adata = anndata.AnnData(X=X,obs=obs)
    adata.var['Labels'] = None

    ## Subsample
    max_size = 100000
    test_size = 1-max_size/len(adata)
    if test_size > 0:
        adata,_ = train_test_split(adata,test_size = test_size,stratify=adata.obs['Labels'])

    sc.pp.scale(adata)

    palette_manual = None

    return(adata,palette_manual)


def prepare_data():

    random.seed(0)

    ImageDims = [28,28,1]

    adata, palette_manual = load_adata()
    adata_f = adata.copy()
    adata_f = reduce_samples(adata_f, label_in = adata.obs['Labels'], type='sample',num_reduced=10000).copy().transpose()
    adata_f.obs['Labels'] = None

    ## Set up data_handlers ====================================================

    ## Data handler per sample
    datah_sample = dh.data_handler(X=adata,y=adata)
    datah_sample.create_split_index_list(k_split=0.8,random_seed=0)
    datah_sample.create_dataset(kfold_idx=0)

    ## Data handler per feature
    datah_feature = dh.data_handler(X=adata_f)

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

    ## Set up Samples
    sample_set = None

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

    return(kwargs_FI, sample_set, datah_feature, datah_sample, plot_args_dict_sf)
