import os

import copy

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

import gc

import scanpy as sc


def load_adata(subsets_labels = None, scale = True, normalized = True, hvgs = None,
               num_hvgs=2000, num_sample=20000, raw=False):

    """ Create annotated data """

    random.seed(0)

    if dataset == 'LargeBrain':
        filename = "data/LargeBrain/expression.h5ad"
    elif dataset == 'BrainCortex'
        filename = "data/BrainCortex/expression.h5ad"

    adata = sc.read_h5ad(filename)

    palette_manual = None

    if subsets_labels is not None:
        adata = adata[np.isin(adata.obs['Labels'],subsets_labels),]

    ## Filter genes
    gc.collect()

    ## Subsample
    test_size = 1-num_sample/len(adata)
    if test_size > 0:
        adata,_ = train_test_split(adata,test_size = test_size,stratify=adata.obs['Labels'])
    gc.collect()

    if adata.X.shape[-1] > num_hvgs:
        adata.X = adata.X[:,:num_hvgs]

    if adata.isview:
        sc._utils.view_to_actual(adata)

    ## Convert sparse matrix to matrix
    try:
        adata.X = np.array(adata.X.toarray())
    except:
        pass

    gc.collect()

    return(adata,palette_manual)


def prepare_data(num_reduced = 10000, sample_size = 100, reduce_mode='sample', reduce_subset = None, **kwargs):

    random.seed(0)

    ImageDims = None

    adata, palette_manual = load_adata(**kwargs)

    ## Set up data_handlers ====================================================

    #### Create data handler object for both sample and feature
    datah_sample  = dh.adata2datah(adata, mode = 'sample')
    datah_feature = dh.adata2datah(adata, mode = 'feature',
                                   num_reduced   = num_reduced,
                                   reduce_mode   = reduce_mode,
                                   reduce_subset = reduce_subset)

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

    hue_order = None

    ## Set Additional parameters
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
                         'ImageDima': ImageDims
                         }

    ## Set up Samples
    if sample_size is not None:
        max_size = sample_size
        test_size = 1-max_size/len(adata)

        if test_size > 0:
            if min(np.unique(adata.obs['Labels'],return_counts=True)[1]) > 1:
                adata_sample,_ = train_test_split(adata, test_size = test_size, stratify=adata.obs['Labels'])
            else:
                adata_sample,_ = train_test_split(adata, test_size = test_size)
            sample_set  = {'labels': adata_sample.obs['Labels'],
                          'samples': adata_sample.X}
        else:
            sample_set = None

    else:
        sample_set = None

    return(kwargs_FI, sample_set, datah_feature, datah_sample, plot_args_dict_sf)
