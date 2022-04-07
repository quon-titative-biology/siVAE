import pyreadr
import scanpy as sc
import pandas as pd

result = pyreadr.read_r('/share/quonlab/workspaces/ruoxinli/multitask_/data/iPSC_neuronal/FPP/regress_cellcycle_expid_selected_cells_genes.rda')
hvgs_rita = result['HVGsel_tmp'].values.reshape(-1)

df = pd.read_csv('data/D11/experiment/FPP/hvg.csv',index_col=0)
hvgs = df.index[np.any(df.values,axis=1)].values
hvgs_all = df.index[df['all']].values

np.intersect1d(hvgs_rita,hvgs)
np.intersect1d(hvgs_rita,hvgs_all)

adata_regressed = sc.read_h5ad('data/D11/experiment/FPP/regressed_out.h5ad')
