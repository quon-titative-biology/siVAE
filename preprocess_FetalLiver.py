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


logdir_ArrayExpress = "/home/yongin/projects/siVAE/data/fetal_liver/ArrayExpress"

filename = os.path.join(logdir_ArrayExpress,"fulldata.h5ad")

if os.path.isfile(filename):
    df_exp = pd.read_csv(os.path.join(logdir_ArrayExpress,'E-MTAB-7407.sdrf.txt'),sep='\t')

    key2value = {'Source Name'                  : 'Source Name',
                 'Characteristics[age]'         : 'Week gestation',
                 'Characteristics[sex]'         : 'Gender',
                 'Characteristics[individual]'  : 'Individual',
                 'Characteristics[facs sorting]': 'facs sorting',
                 'Factor Value[organism part]'  : 'Organ',
                 'Factor Value[facs sorting]'   : 'facs sorting value'}

    adata = None
    for source in df_exp['Source Name']:
        logdir_source = os.path.join(logdir_ArrayExpress,source)
        if os.path.exists(logdir_source):
            print(source)
            try:
                adata_new = sc.read_10x_mtx(os.path.join(logdir_source,'GRCh38'))
                df = pd.read_csv(os.path.join(logdir_source,source+'.csv'))
                metadata = df_exp[df_exp['Source Name'] == source]
                for key,value in key2value.items():
                    df[value] = metadata[key].iloc[0]
                adata_new.obs = df
                adata_new.obs_names = adata_new.obs['Cell.Labels']
                if adata is None:
                    adata = adata_new
                else:
                    adata = adata.concatenate(adata_new)
            except:
                pass
    adata.obs['Cell.Labels'] = adata.obs['Cell.Labels'].astype('str')
    adata.obs['Labels'] = adata.obs['Cell.Labels']
    adata.obs_names = range(len(adata))
    adata.write(filename)

else:

    adata = sc.read_h5ad(filename)

do_plot = True

## Remove two sources that is difficult to integrate
subset_sourcename = ["FCAImmP7316901","FCAImmP7352196"]
adata = adata[np.invert(np.isin(adata.obs['Source Name'],subset_sourcename)),]

logdir = os.path.join(logdir_ArrayExpress,'filtered')
os.makedirs(logdir,exist_ok =True)

## Filter genes
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

## Mitochondrial genes
adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

if do_plot:
    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                 jitter=0.4, multi_panel=True)
    plt.savefig(os.path.join(logdir,"mito_pct.pdf"))
    plt.close()

    sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
    plt.savefig(os.path.join(logdir,"total_vs_pt_mt.pdf"))
    plt.close()

    sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')
    plt.savefig(os.path.join(logdir,"total_vs_n_genes.pdf"))
    plt.close()

adata = adata[adata.obs.n_genes_by_counts < 8000, :]
adata = adata[adata.obs.pct_counts_mt < 20, :]

## log-normalize
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

## HVGs
sc.pp.highly_variable_genes(adata, n_top_genes = 5000)
sc.pl.highly_variable_genes(adata)
plt.savefig(os.path.join(logdir,"hvg.pdf"))
plt.close()

adata = adata[:, adata.var.highly_variable]

## Save
filename = os.path.join(logdir,'adata.h5ad')
adata.write(filename)

## PCA
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack')
sc.pl.pca(adata, color='Cell.Labels')
plt.savefig(os.path.join(logdir,"pca.pdf"))
plt.close()

sc.pl.pca_variance_ratio(adata, log=True)
plt.savefig(os.path.join(logdir,"pca_var.pdf"))
plt.close()

## UMAP
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

sc.tl.umap(adata)
sc.pl.umap(adata, color='Cell.Labels', use_raw = False)
plt.savefig(os.path.join(logdir,"umap.pdf"))
plt.close()

## Plot PCA subset
subsets = []
subsets_labels = ['HSC/MPP','MEMP','Mast cell', 'Early Erythroid', 'Mid  Erythroid', 'Late Erythroid']
subsets.append(subsets_labels)
subsets_labels = ['HSC/MPP','MEMP','pre pro B cell ', "Neutrophil-myeloid progenitor"]
subsets.append(subsets_labels)
subsets_labels = ['HSC/MPP', 'Monocyte', 'Mono-Mac', 'Kupffer Cell', "Neutrophil-myeloid progenitor"]
subsets.append(subsets_labels)
subsets_labels = ['HSC/MPP', 'pre pro B cell ', 'pro B cell', 'pre B cell', 'B cell']
subsets.append(subsets_labels)

for ii,subset in enumerate(subsets):
    adata_subset = adata[np.isin(adata.obs['Labels'],subset)]
    sc.pp.filter_genes(adata_subset, min_cells=3)
    sc.pp.scale(adata_subset, max_value=10)
    sc.tl.pca(adata_subset, svd_solver='arpack')
    sc.pl.pca(adata_subset, color='Cell.Labels')
    plt.savefig(os.path.join(logdir,"pca_subset_{}.pdf".format(ii)))
    plt.close()
