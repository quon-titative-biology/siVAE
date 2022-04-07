## System
import os,gc
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'  # no debugging from TF
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

## Plots
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

import scanpy as sc

## siVAE
# Util
from siVAE import util
from siVAE.util import reduce_dimensions

logdir = 'out/exp2/iPSC_neuronal/batch/128'
datadir='data/iPSC_neuronal/D11_scaled_sampled.h5ad'

adata= sc.read_h5ad(datadir)

result_dict = {}
for batch in ['True','False']:
    siVAE_result_dir = os.path.join(logdir,batch,'kfold-0','siVAE_result.pickle')
    siVAE_result = util.load_pickle(siVAE_result_dir)
    result_dict[batch] = siVAE_result
    for mode in ['VAE','siVAE']:
        if mode == 'siVAE':
            adata_plot = sc.AnnData(X=siVAE_result.get_sample_embeddings(), obs=adata.obs)
        elif mode == 'VAE':
            adata_plot = sc.AnnData(X=siVAE_result['sample'].get_value('z_mu'), obs=adata.obs)
        #
        sc.pp.scale(adata_plot)
        #
        sc.tl.pca(adata_plot, svd_solver='arpack')
        sc.pp.neighbors(adata_plot,n_neighbors=5,)
        sc.tl.umap(adata_plot)
        ## plots
        color = ['celltype', 'cluster_id', 'sample_index', 'pool_id']
        # umap
        sc.pl.umap(adata_plot, color=color)
        plt.savefig(os.path.join(logdir, f'umap-{batch}-{mode}.svg'))
        plt.close()
        # pca
        sc.pl.pca(adata_plot, color=color)
        plt.savefig(os.path.join(logdir, f'pca-{batch}-{mode}.svg'))
        plt.close()
