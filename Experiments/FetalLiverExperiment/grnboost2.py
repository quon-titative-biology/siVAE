import os
import time

import pandas as pd
import numpy as np

from arboreto.utils import load_tf_names
from arboreto.algo import grnboost2

from sklearn.preprocessing import scale

## Plots
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

def run_GRNBOOST2(ex_matrix,outdir,genes=None,cells=None,do_scale=True):
    """
    Input: ex_matrix, cell x gene numpy array/matrix with index as gene names and columns as cells
    """
    start_time = time.time()
    ## Set up data frme
    ex_matrix = ex_matrix.transpose()
    if do_scale: ex_matrix = scale(ex_matrix,1)
    if genes is None: genes = ['G'+str(ii+1) for ii in np.arange(ex_matrix.shape[0])]
    ex_matrix = pd.DataFrame(ex_matrix,
                             index= genes,
                             columns=cells)
    GRNnetwork = grnboost2(expression_data=ex_matrix.transpose(),
                           verbose = True)
    total_time = time.time() - start_time
    print(total_time)
    ##
    genes = ex_matrix.index
    cells = ex_matrix.columns
    ngene = len(genes)
    ## Create adjacency matrix
    gene2idx = {gene:ii for ii, gene in enumerate(genes)}
    adj_mat = np.zeros([ngene, ngene])
    for edge in GRNnetwork.values:
        g1, g2, value = edge
        adj_mat[gene2idx[g1],gene2idx[g2]] = value
    ## Normalize by the number of samples
    adj_mat = adj_mat / ex_matrix.shape[1]
    ## Save to csv
    os.makedirs(outdir,exist_ok=True)
    csvfile = os.path.join(outdir,'adjmat.csv')
    df_result = pd.DataFrame(adj_mat,index=genes,columns=genes)
    df_result.to_csv(csvfile, index = True, header = True)
    ## Plot
    sns.heatmap(data = df_result,cmap='Blues')
    plt.savefig(os.path.join(outdir,'adjmat.pdf'))
    plt.close()
    #
    df_genes = df_result.sum(1)
    csvfile = os.path.join(outdir,'adjmat_genes.csv')
    df_genes.to_csv(csvfile, index = True, header = ['Connectivity'])

if __name__ == "__main__":

    ## Perform GRNboost2 on raw data
    datadir = 'out/data/siVAE'
    outdir = 'out/GRN'
    os.makedirs(outdir,exist_ok=True)

    list_npz = [npz.split('.npz')[0] for npz in os.listdir(datadir)]

    for input_type in list_npz:

        filename = os.path.join(datadir, input_type+'.npz')
        logdir_result = os.path.join(outdir,input_type,'GRNBOOST2')
        ## Load npz
        npz = np.load(filename,allow_pickle=True)
        exprs = npz['exprs'] # gene x cell
        genes = npz['gene_names']
        cells = npz['cell_names']
        ## Run GRNBOOST2
        run_GRNBOOST2(ex_matrix = exprs,
                      outdir    = logdir_result,
                      genes     = genes,
                      cells     = cells,
                      do_scale  = True)
