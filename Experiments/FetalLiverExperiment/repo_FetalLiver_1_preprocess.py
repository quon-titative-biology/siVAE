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


## Data were downloaded from https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-7407/

## Set the directory to where data is located
logdir_data = 'data'
logdir_ArrayExpress = os.path.join("data/ArrayExpress")

filename = os.path.join(logdir_data,"fulldata.h5ad")

#### ===========================================================================
#### Import count data and metadata as AnnData
#### ===========================================================================

## If combined fetal liver dataset does not exist in h5ad format, read through
## the data downloaded from source then convert too AnnData with scanpy
if not os.path.isfile(filename):

    ## Upload and rename per-source metadata
    df_metadata_source = pd.read_csv(os.path.join(logdir_data,'E-MTAB-7407.sdrf.txt'),sep='\t')

    ## Set mapping to rename the metadata category
    key2value = {'Source Name'                  : 'Source Name',
                 'Characteristics[age]'         : 'Week gestation',
                 'Characteristics[sex]'         : 'Gender',
                 'Characteristics[individual]'  : 'Individual',
                 'Characteristics[facs sorting]': 'facs sorting',
                 'Factor Value[organism part]'  : 'Organ',
                 'Factor Value[facs sorting]'   : 'facs sorting value'}

    adata = None

    ## Iterate through sources and concatenate expression data to AnnData
    for source in df_metadata_source['Source Name']:

        logdir_source = os.path.join(logdir_ArrayExpress,source)

        if os.path.exists(logdir_source):
            print(source)

            try:

                ## Read expression data
                adata_new = sc.read_10x_mtx(os.path.join(logdir_source,'GRCh38'))

                ## Read per-cell metadata
                df_metadata = pd.read_csv(os.path.join(logdir_source,source+'.csv'))

                ## Isolate per-source metadata and update per-cell metadata
                metadata = df_metadata_source[df_metadata_source['Source Name'] == source]

                for key,value in key2value.items():
                    df_metadata[value] = metadata[key].iloc[0]

                ## Create and concatenate new AnnData
                adata_new.obs = df_metadata
                adata_new.obs_names = adata_new.obs['Cell.Labels']

                if adata is None:
                    adata = adata_new
                else:
                    adata = adata.concatenate(adata_new)

            except:
                pass

    ## Modify label format and save AnnData
    adata.obs['Cell.Labels'] = adata.obs['Cell.Labels'].astype('str')
    adata.obs['Labels'] = adata.obs['Cell.Labels']
    adata.obs_names = range(len(adata))
    adata.write(filename)

else:
    ## Load saved AnnData
    adata = sc.read_h5ad(filename)

#### ===========================================================================
#### Pre-processing
#### ===========================================================================


logdir = os.path.join(logdir_data)
os.makedirs(logdir,exist_ok =True)

## Save raw data for models that takes in counts
adata.raw = adata

## Set True to plot pre-processing metrics
do_plot = False

## Remove two sources that is difficult to integrate
subset_sourcename = ["FCAImmP7316901","FCAImmP7352196"]
idx_keep = np.invert(np.isin(adata.obs['Source Name'],subset_sourcename))
adata = adata[idx_keep]

## Filter genes
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

## Calculate and plot qc metrics
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

    for metric in ['mean_counts','total_counts','pct_dropout_by_counts', 'n_cells']:

        ax = sns.violinplot(x=adata.var[metric],orient='v')
        plt.savefig(os.path.join(logdir,"QC-genes-violin-{}.svg".format(metric)))
        plt.close()

        ax = sns.distplot(adata.var[metric],kde=False)
        plt.savefig(os.path.join(logdir,"QC-genes-dist-{}.svg".format(metric)))
        plt.close()

## Filter out genes based on the above metrisc
adata = adata[adata.obs.n_genes_by_counts < 8000, :]
adata = adata[adata.obs.pct_counts_mt < 20, :]

## log-normalize
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

## Identify highly variable genes
sc.pp.highly_variable_genes(adata, n_top_genes = 5000)
sc.pl.highly_variable_genes(adata)
plt.savefig(os.path.join(logdir,"hvg.pdf"))
plt.close()

adata = adata[:, adata.var.highly_variable]

## Save
filename = os.path.join(logdir,'adata_with_raw.h5ad')
adata.write(filename)

adata = sc.read_h5ad(filename)

## Temporarily create a sample version for testing
# subsets_labels = ['Hepatocyte','Kupffer Cell','NK','Mono-NK','Mac NK',
#                   'pro B cell','pre B cell','pre pro B cell']
# adata_temp = adata[np.isin(adata.obs['Labels'],subsets_labels),]
#
# filename = os.path.join(logdir,'adata_with_raw_temp.h5ad')
# adata_temp.write(filename)

#### ===========================================================================
#### Visualize the preprocessed data with PCA and t-SNE
#### ===========================================================================

if do_plot:
    #### Plot PCA
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pl.pca(adata, color='Cell.Labels')
    plt.savefig(os.path.join(logdir,"pca.pdf"))
    plt.close()

    sc.pl.pca_variance_ratio(adata, log=True)
    plt.savefig(os.path.join(logdir,"pca_var.pdf"))
    plt.close()

    #### Plot UMAP
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

    sc.tl.umap(adata)
    sc.pl.umap(adata, color='Cell.Labels', use_raw = False)
    plt.savefig(os.path.join(logdir,"umap.pdf"))
    plt.close()

    #### Plot PCA subset
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
