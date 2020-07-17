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

def load_adata():
    """ Create annotated data """
    #
    adata = sc.read_h5ad(filename)
    #
    return(adata)

def prepare_data(architecture, LE_dim):

    random.seed(0)

    adata = load_adata()

    palette_manual = None

    ## Subset by types
    # All
    subsets_labels = None

    ## Subsample
    test_size = 1-20000/len(adata)
    if test_size > 0:
        adata,_ = train_test_split(adata,test_size = test_size,stratify=adata.obs_names)

    ImageDims = None

    sc.pp.scale(adata)

    adata_f = adata.copy()
    adata_f = adata.transpose()
    adata_f.obs['Labels'] = None

    # ==============================================================================

    #### Set up data_handlers

    ## Data handler per sample
    datah_sample = dh.data_handler(X=adata,y=adata)
    datah_sample.create_split_index_list(k_split=0.8,random_seed=0)
    datah_sample.create_dataset(kfold_idx=0)

    ## Data handler per feature
    datah_feature = dh.data_handler(X=adata_f)

    ## Set up architecture of hidden layers
    h_dims = architecture.replace('LE',str(LE_dim))
    h_dims = [int(dim) for dim in h_dims.split("-")]
    datah_sample.index_latent_embedding = int(h_dims.pop())
    datah_sample.h_dims = h_dims

    ## For datah_feature
    h_dims = architecture.replace('LE',str(LE_dim))
    h_dims = [int(dim) for dim in h_dims.split("-")]
    datah_feature.index_latent_embedding = int(h_dims.pop())
    _ = h_dims.pop(-2)
    datah_feature.h_dims = h_dims

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

    hue_order = ['EPCAM and cholangiocytes','Hepatocytes','Kupffer cells',
                 'Stellate cells','NA']

    ## Set Additional parameters
    data_name = 'HumanLiver'

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

    plot_args_dict_sf = {'feature': plot_args_feature,
                         'sample': plot_args_sample}

    if palette_manual is not None:
        palette = palette_manual

    return(kwargs_FI, sample_set, datah_feature, datah_sample, palette, plot_args_dict_sf, ImageDims, data_name, hue_order)
