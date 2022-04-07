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


import scanpy as sc

def variance(adata=None,X=None, n_batch=None):
    import gc
    if n_batch is None:
        if adata is not None:
            X = adata.X
        if type(X) == np.ndarray:
            var = np.square(X.std(0))
        else:
            var = np.array(X.power(2).mean(0) - np.power(X.mean(0),2))[0]
    else:
        var_ii = []
        for ii in range(int(adata.shape[-1]/n_batch)):
            i1 = ii * n_batch
            i2 = (ii+1) * n_batch
            X_ii = adata.X[:,i1:i2]
            var_ii.append(variance(X=X_ii))
            gc.collect()
        var = np.concatenate(var_ii)
        gc.collect()
    return var


def load_adata(subsets_labels = None, scale = True, normalized = True, hvgs = None,
               num_hvgs=2000, num_sample=20000, raw=False):

    """ Create annotated data """

    random.seed(0)

    if normalized:
        filename = "data/adata_with_raw.h5ad"
    else:
        filename = "data/fulldata.h5ad"

    adata = sc.read_h5ad(filename)
    adata.obs['Cell Type'] = adata.obs['Labels']
    adata = adata[np.invert(adata.obs['Labels'] == 'nan')]

    palette_manual = None

    if subsets_labels is not None:
        adata = adata[np.isin(adata.obs['Labels'],subsets_labels),]

    ## Filter genes
    # sc.pp.filter_genes(adata, min_cells=3)
    gc.collect()

    if hvgs is not None:
        adata.var['highly_variable'] = np.isin(adata.var['Labels'],hvgs)
    else:
        var = variance(adata=adata,n_batch=2000)
        idx = np.argsort(var)[::-1][:num_hvgs]
        adata.var['highly_variable'] = np.isin(np.arange(adata.shape[-1]),idx)

    adata = adata[:,adata.var.highly_variable]
    gc.collect()

    ## Subsample
    test_size = 1-num_sample/len(adata)
    if test_size > 0:
        adata,_ = train_test_split(adata,test_size = test_size,stratify=adata.obs['Labels'])
    gc.collect()

    if scale:
        sc.pp.scale(adata)

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

    ## Add addtional labels based on GSEA genesets
    cell_labels = adata.obs['Labels']
    cell_labels_dict = {k:k for k in np.unique(cell_labels)}
    cell_labels_dict['Hepatocyte']      = 'Hepatocytes'
    cell_labels_dict['Kupffer Cell']    = 'Kupffer_Cells'
    cell_labels_dict['NK']              = 'NK_NKT_cells'
    cell_labels_dict['Mono-NK']         = 'NK_NKT_cells'
    cell_labels_dict['Mac NK']          = 'NK_NKT_cells'
    cell_labels_dict['Fibroblast']      = 'Stellate_cells'
    cell_labels_dict['HSC/MPP']         = 'Stellate_cells'
    cell_labels_dict['pro B cell']      = 'MHC_II_pos_B'
    cell_labels_dict['pre B cell']      = 'MHC_II_pos_B'
    cell_labels_dict['pre pro B cell']  = 'MHC_II_pos_B'
    cell_labels_2 = [cell_labels_dict[c] for c in cell_labels]
    adata.obs['GSEA label'] = cell_labels_2

    ## Map reduce subset from string to list

    GSLabel2DSLabel = {}
    for k,v in cell_labels_dict.items():
        GSLabel2DSLabel.setdefault(v, []).append(k)

    if reduce_subset == 'All':
        reduce_subset = None
    else:
        reduce_subset = GSLabel2DSLabel[reduce_subset]

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

    ## Set manual palette
    palette['HSC/MPP'] = "lemonchiffon"

    palette['pre pro B cell '] = "royalblue"
    palette['pro B cell']      = 'blue'
    palette['pre B cell']      = 'navy'
    palette['B cell']          = 'darkblue'

    palette['ILC precursor']  = 'chartreuse'
    palette['Early L/TL']     = 'palegreen'
    palette['NK']             =  'darkgreen'

    palette["Neutrophil-myeloid progenitor"] = 'grey'

    palette['pDC precursor'] = 'mistyrose'
    palette['Monocyte-DC precursor']  = 'salmon'
    palette['DC1']           = 'lightcoral'
    palette['DC2']           = 'maroon'

    palette['Monocyte precursor'] = 'pink'
    palette['Monocyte']           = 'hotpink'
    palette['Mono-Mac']           = 'magenta'
    palette['Kupffer Cell']       = 'purple'

    palette['VCAM1+ Erythroblastic Island Macrophage'] = 'teal'
    palette['Mast cell'] = 'cyan'
    palette['MEMP'] = 'darkcyan'

    palette['Megakaryocyte']   = 'peachpuff'
    palette['Early Erythroid'] = 'darkorange'
    palette['Mid  Erythroid']   = 'navajowhite'
    palette['Late Erythroid']  = 'orange'

    palette['Endothelial cell'] = 'brown'
    palette['Fibroblast']       = 'saddlebrown'
    palette['Hepatocyte']       = 'mediumslateblue'

    palette['NA'] = 'lightgrey'

    hue_order = ['HSC/MPP',
                 'pre pro B cell ', 'pro B cell', 'pre B cell', 'B cell',
                 'ILC precursor', 'Early L/TL', 'NK', "Neutrophil-myeloid progenitor",
                 'pDC precursor', 'Monocyte-DC precursor', 'DC1', 'DC2',
                 'Monocyte precursor', 'Monocyte', 'Mono-Mac', 'Kupffer Cell',
                 'VCAM1+ Erythroblastic Island Macrophage', 'Mast cell','MEMP',
                 'Megakaryocyte', 'Early Erythroid', 'Mid  Erythroid', 'Late Erythroid',
                 'Endothelial cell', 'Fibroblast', 'Hepatocyte', 'NA']

    hue_order = np.array(hue_order)[np.isin(hue_order,keys)]
    hue_order = np.concatenate([hue_order,keys[np.invert(np.isin(keys,hue_order))]])

    ## Set Additional parameters
    data_name = 'FetalLiver'

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
