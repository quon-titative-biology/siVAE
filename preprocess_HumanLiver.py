import os

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

# Plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns


logdir = "/home/yongin/projects/siVAE/data/HumanLiver"

filename = os.path.join(logdir,"fulldata.h5ad")

do_plot = False

keys = {11:'Hepatocytes',
        17:'Hepatocytes',
        14:'Hepatocytes',
        21:'Stellate cells',
        4 :'EPCAM and cholangiocytes',
        7 :'EPCAM and cholangiocytes',
        39:'EPCAM and cholangiocytes',
        24:'EPCAM and cholangiocytes',
        31:'Kupffer cells',
        23:'Kupffer cells',
        6 :'Kupffer cells',
        25:'Kupffer cells',
        2 :'Kupffer cells',
        20:'Liver sinusoidal endothelial cells',
        9 :'Liver sinusoidal endothelial cells',
        13:'Liver sinusoidal endothelial cells',
        29:'Mascrovascular endothelial cells',
        10:'Mascrovascular endothelial cells'}


if os.path.isfile(filename):

    df = pd.read_csv(os.path.join(logdir,'GSE124395_Normalhumanlivercellatlasdata.txt'),sep='\t')
    metadata = pd.read_csv(os.path.join(logdir,'GSE124395_clusterpartition.txt'),sep=" ")
    metadata["Labels"] = metadata.index
    metadata.columns = ['Cluster','Name']

    df = df[metadata['Name']]
    adata = sc.AnnData(df.transpose(),obs = metadata)
    adata.write(filename)

else

    adata = sc.read_h5ad(filename)

for ii in range(adata.obs['Cluster'].max()):
    if not ii in keys.keys():
        keys[ii] = 'None'

adata.obs['Labels'] = adata.obs['Cluster'].map(keys)
subsets = ['Hepatocytes','Stellate cells','EPCAM and cholangiocytes','Kupffer cells']
adata = adata[np.isin(adata.obs['Labels'],subsets)]

## Filter cells and genes
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

## Mitochondrial
adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
if do_plot:
    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                 jitter=0.4, multi_panel=True)
    plt.savefig(os.path.join(logdir_ArrayExpress,"mito_pct.pdf"))
    plt.close()

    sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
    plt.savefig(os.path.join(logdir_ArrayExpress,"total_vs_pt_mt.pdf"))
    plt.close()

    sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')
    plt.savefig(os.path.join(logdir_ArrayExpress,"total_vs_n_genes.pdf"))
    plt.close()

adata = adata[adata.obs.n_genes_by_counts < 8000, :]
adata = adata[adata.obs.pct_counts_mt < 20, :]

## Normalize
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

## HVGs
sc.pp.highly_variable_genes(adata, n_top_genes = 2000)
adata = adata[:, adata.var.highly_variable]
if do_plot:
    sc.pl.highly_variable_genes(adata)
    plt.savefig(os.path.join(logdir_ArrayExpress,"hvg.pdf"))
    plt.close()
