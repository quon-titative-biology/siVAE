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

def load_adata(subsets_labels = None):

    """ Create annotated data """

    random.seed(0)

    filename = "/home/yongin/projects/siVAE/data/fetal_liver/ArrayExpress/filtered/adata.h5ad"
    adata = sc.read_h5ad(filename)
    adata.obs['Cell Type'] = adata.obs['Labels']
    adata = adata[np.invert(adata.obs['Labels'] == 'nan')]

    # npz = np.load('data/fetal_liver/fetal_liver.npz', allow_pickle=True)
    # X      = npz['exprmat'].transpose()
    # metadata = npz['metadata']
    # metadata_name = npz['metadata_names']
    # metadata_name[0] = 'Cell Type'
    # labels = metadata[:,0]
    # labels_gene  = npz['genes'][:,0]
    # metadata = pd.DataFrame(metadata,columns=metadata_name)
    # adata = sc.AnnData(X, obs=metadata, var=labels_gene)
    # adata.obs['Labels'] = adata.obs['Cell Type']
    # adata.var['Labels'] = adata.var[0]

    palette_manual = None

    ## Subset by types
    # 1
    subsets_labels = ['HSC/MPP','MEMP','Mast cell', 'Early Erythroid', 'Mid  Erythroid', 'Late Erythroid']
    palette_manual = {'HSC/MPP': 'grey',
               'MEMP': 'palegreen',
               'Mast cell': 'deepskyblue',
               'Early Erythroid': 'bisque',
               'Mid  Erythroid' : 'sandybrown',
               'Late Erythroid' : 'sienna',
               'NA': 'black'}

    # # 2
    # subsets_labels = ['HSC/MPP','MEMP','pre pro B cell ', "Neutrophil-myeloid progenitor"]

    # # 3
    # subsets_labels = ['HSC/MPP', 'Monocyte', 'Mono-Mac', 'Kupffer Cell', "Neutrophil-myeloid progenitor"]
    # palette_manual = {}
    # palette_manual['Monocyte precursor'] = 'pink'
    # palette_manual['Monocyte']           = 'hotpink'
    # palette_manual['Mono-Mac']           = 'magenta'
    # palette_manual['Kupffer Cell']       = 'purple'
    # palette_manual['NA'] = 'black'
    # palette_manual["Neutrophil-myeloid progenitor"] = 'deepskyblue'
    # palette_manual['HSC/MPP'] = 'grey'

    ## 4
    # subsets_labels = ['HSC/MPP', 'pre pro B cell ', 'pro B cell', 'pre B cell', 'B cell']

    # 5
    # subsets_labels = ['HSC/MPP']
    # palette_manual = {'HSC/MPP': 'grey',
    #                   'NA': 'black'}

    if subsets_labels is not None:
        adata = adata[np.isin(adata.obs['Labels'],subsets_labels),]

    ## Subsample
    test_size = 1-20000/len(adata)
    if test_size > 0:
        adata,_ = train_test_split(adata,test_size = test_size,stratify=adata.obs['Labels'])

    ## Filter genes
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.highly_variable_genes(adata, n_top_genes = 2000)
    adata = adata[:, adata.var.highly_variable]

    adata_t = adata.copy().transpose()
    adata = adata_t.concatenate([adata_t]*13).copy().transpose()

    # sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.scale(adata)

    #
    return(adata,palette_manual)


def prepare_data():

    random.seed(0)

    ImageDims = None

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
                         'ImageDima': ImageDims
                         }

    return(kwargs_FI, sample_set, datah_feature, datah_sample, plot_args_dict_sf)
