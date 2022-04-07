import os,gc

import scanpy as sc
import scanpy.external as sce
import anndata

import numpy as np
import pandas as pd

## Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

adata = anndata.read_h5ad('D11.h5')

logdir = "data/D11"
os.makedirs(logdir,exist_ok=True)


def variance(adata=None,X=None, n_batch=None):
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

# ==============================================================================
#                                 Pre-processing
# ==============================================================================

# -----------------------------------
# Import differential efficiency data
# -----------------------------------

df_eff = pd.read_csv('diff_efficiency_neur.csv')
df_eff.index = df_eff.donor_id
donor_overlap = np.intersect1d(df_eff.donor_id,adata.obs.donor_id)

df_eff = df_eff.sort_values('diff_efficiency')

sns.histplot(df_eff, x='in_study', hue = 'in_study')
plt.savefig(os.path.join(logdir,'line_state.svg'))
plt.close()

## Remove lines not in differential efficiency data or
donor_keep = df_eff[df_eff.in_study != 'not_assessed'].donor_id.to_numpy()
adata = adata[np.isin(adata.obs.donor_id,donor_keep)]

# ----------------------------------------------------
# Read cell cycle genes (snippet from scanpy tutorial)
# ----------------------------------------------------

cell_cycle_genes = [x.strip() for x in open('data/regev_lab_cell_cycle_genes.txt')]
s_genes = cell_cycle_genes[:43]
g2m_genes = cell_cycle_genes[43:]
cell_cycle_genes = [x for x in cell_cycle_genes if x in adata.var_names]

# ---------------------------
# Perform basic qc operations
# ---------------------------

## Already normalized data
adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

for cat in ['n_genes_by_counts', 'total_counts', 'pct_counts_mt']:
    sc.pl.violin(adata, cat,
                 jitter=0.4, multi_panel=True)
    plt.savefig(os.path.join(logdir,f'violin_{cat}.svg'))
    plt.close()

sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
plt.savefig(os.path.join(logdir,'counts_vs_mt.svg'))
plt.close()

sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')
plt.savefig(os.path.join(logdir,'totalcounts_vs_genes_by_counts.svg'))
plt.close()

## Only normalized, need to perform log1p
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

adata.write(os.path.join(logdir,'D11_normalized.h5ad'))


#### Measure correlation betewen MT genes and efficiencies
# adata_subset = adata[adata.obs['celltype'] == 'FPP',adata.var.mt]
top_genes = ['MT-ND3','MT-ATP6','RPS27','MT-CYB','MT-CO1']
top_genes = ['MT-ND3','MT-ATP6','S100A11','MT-CYB','MT-CO1']
idx = np.array([np.where(g==adata.var_names)[0] for g in top_genes]).reshape(-1)
idx = np.isin(adata.var_names,adata_sample.var_names)
adata_subset = adata[:,idx]
# adata_subset = adata

df_exp = pd.DataFrame(adata_subset.X.toarray())
df_exp['donor_id'] = adata_subset.obs['donor_id'].to_list()
df_mean = df_exp.groupby('donor_id').mean()

from scipy.stats import spearmanr
eff = df_eff.loc[df_mean.index]['diff_efficiency']

df_corr = pd.DataFrame([list(spearmanr(exp,eff)) for gene,exp in df_mean.iteritems()],
                       index=adata_subset.var_names,
                       columns=['SpearmanCorr','P-value'])
df_corr['P-adj'] = df_corr['P-value'] * df_corr.shape[0]
df_corr.to_csv(os.path.join(logdir,'corr_eff_meanexp.csv'))

df_mean.columns = adata_subset.var_names
df_mean.index = eff
df_plot = df_mean.unstack().reset_index()
df_plot.columns = ['Task','Efficiency','Mean Expression']

g = sns.lmplot(data=df_plot,x='Efficiency',y='Mean Expression',hue='Task',ci=None)
plt.savefig(os.path.join(logdir,'Eff vs DC.svg'))
plt.close()


#### Save the filtered data
var = variance(adata)
rank_var = np.argsort(var)[::-1][:2000]
highly_variable = np.isin(np.arange(adata.shape[-1]),rank_var)
adata.var['highly_variable'] = highly_variable
adata = adata[:,adata.var['highly_variable']]

## Binarize sample_index
def convert2onehot(X):
    n_value = len(np.unique(X))
    return np.eye(n_value)[X]

cats = adata.obs['pool_id'].cat.codes
adata.obsm['batch'] = convert2onehot(cats)

## Visualize

# subset for experiment
from sklearn.model_selection import train_test_split
adata_train,_ = train_test_split(adata,train_size=0.05)

sc.pp.scale(adata_train)

sc.tl.pca(adata_train, svd_solver='arpack')
sc.pp.neighbors(adata_train,n_neighbors=5,)
sc.tl.umap(adata_train)
## plots
color = ['celltype', 'cluster_id', 'sample_index', 'pool_id']
# umap
sc.pl.umap(adata_train, color=color)
plt.savefig(os.path.join(logdir, 'umap-sample.svg'))
plt.close()
# pca
sc.pl.pca(adata_train, color=color)
plt.savefig(os.path.join(logdir, 'pca-sample.svg'))
plt.close()

adata_train.write(os.path.join(logdir,'D11_scaled_sampled.h5ad'))



## Plot the correlation between percentage of progenitor cells

# matrix cell lines x cell types where each value is number of cells
value_counts = pd.DataFrame(adata.obs.value_counts(['donor_id','celltype'])).unstack()
value_counts.columns = [c[1] for c in value_counts.columns.to_list()]
value_percent = value_counts.copy()
value_counts['efficiency'] = df_eff.reindex(value_counts.index).diff_efficiency.values.astype('float')
value_counts['DA efficiency'] = df_eff.reindex(value_counts.index).DA_efficiency.values.astype('float')
value_counts['Sert efficiency'] = df_eff.reindex(value_counts.index).Sert_efficiency.values.astype('float')

# Plot P_FPP total counts per cell line vs efficiency
fig,axes = plt.subplots(ncols=3, figsize=(15,4))
for ax, eff_type in zip(axes,['efficiency','DA efficiency', 'Sert efficiency']):
    g = sns.scatterplot(data=value_counts,x='P_FPP',y=eff_type, ax=ax)
    _ = ax.set_aspect('auto')

plt.savefig(os.path.join(logdir,'Counts vs eff.svg'))
plt.close()

# Plot P_FPP percentage per cell line vs efficiency
value_percent = value_percent.div(value_percent.sum(1), axis=0) # normalize each cell line by total counts
value_percent['efficiency'] = df_eff.reindex(value_percent.index).diff_efficiency.values.astype('float')
value_percent['DA efficiency'] = df_eff.reindex(value_percent.index).DA_efficiency.values.astype('float')
value_percent['Sert efficiency'] = df_eff.reindex(value_percent.index).Sert_efficiency.values.astype('float')

# Plot P_FPP total counts per cell line vs efficiency
fig,axes = plt.subplots(ncols=3, figsize=(15,4))
for ax, eff_type in zip(axes,['efficiency','DA efficiency', 'Sert efficiency']):
    g = sns.scatterplot(data=value_percent,x='P_FPP',y=eff_type, ax=ax)

plt.savefig(os.path.join(logdir,'CountPercent vs eff.svg'))
plt.close()

## Plot number of samples per line
line_counts = adata.obs.donor_id.value_counts()
plt.bar(x=range(len(line_counts)), height=line_counts)
plt.savefig(os.path.join(logdir,'n_sample_per_line.svg'))
plt.close()

# Plot total number of counts per line vs efficiency
df_plot = pd.DataFrame({"donor_id"   : line_counts.index,
                        'Counts'     : line_counts.values,
                        'Efficiency' : df_eff.reindex(line_counts.index).diff_efficiency.values})

sns.scatterplot(data = df_plot, x = 'Counts', y = 'Efficiency')
plt.savefig(os.path.join(logdir,'Lines-Counts_vs_Efficiency.svg'))
plt.close()

# ## Set number of lines to keep
# df_lines_keep = df_plot[(df_plot.Counts > 1000) & (df_plot.Counts < 8000)]
# donor_keep = df_lines_keep.donor_id.values

# -----------------------------------
# Subset adata with selected donor_id
# -----------------------------------
adata = adata[np.isin(adata.obs.donor_id,donor_keep)]

value_counts = adata.obs['donor_id'].value_counts()
value_counts = value_counts[value_counts != 0]
adata.obs['donor_id'] = pd.Categorical(adata.obs['donor_id'],
                                       value_counts.index.astype('str'))

ax = sns.histplot(adata.obs,
                  x = 'donor_id',
                  hue = 'celltype',
                  multiple='stack')
plt.savefig(os.path.join(logdir,'Lines_count_by_ct.svg'))
plt.close()

# --------------------------
# Regressing + HVG selection
# --------------------------

num_hvgs = 3000
num_hvgs_line = 1000

n_jobs = 10

adata.raw = adata

# Perform per experiment
for ct in ['P_FPP','FPP']:

    logdir_exp = os.path.join(logdir, 'experiment', ct)
    os.makedirs(logdir_exp, exist_ok=True)

    # -------------
    # Regress out
    # -------------

    adata_dir_regressed = os.path.join(logdir_exp,'regressed_out.h5ad')

    # Subset to only cell lines with at least 200 cells
    adata_ct = adata[adata.obs.celltype == ct]

    value_counts = adata_ct.obs['donor_id'].value_counts()
    value_counts = value_counts[value_counts != 0]
    donor_keep_ct = value_counts[value_counts > 200].index

    adata_ct = adata_ct[np.isin(adata_ct.obs.donor_id,donor_keep_ct)]

    # ---------------------------------------------------------
    # Generate txt files for creating list of cell lines to run
    # ---------------------------------------------------------
    # Create dataframe for all valid cell lines and concat with df_eff
    df_eff_ct = adata_ct.obs.donor_id.value_counts()
    df_eff_ct.columns='Count'
    df_eff_ct = pd.concat([df_eff_ct, df_eff.reindex(df_eff_ct.index)],axis=1)

    # Per efficiency type, generate text file for top and bottom 5 cell lines
    for eff_type in [c for c in df_eff_ct.columns if 'efficiency' in c]:
        eff = df_eff_ct[eff_type].sort_values()
        lines_exp = np.concatenate([eff[:5].index,eff[-5:].index])
        np.savetxt(os.path.join(logdir_exp,f'lines-{eff_type}.txt'),
                   lines_exp, fmt="%s")

    # ---------------------------------------------------------
    # Run regression then save as h5ad
    # ---------------------------------------------------------

    if os.path.isfile(adata_dir_regressed) and True:
        adata_reg = anndata.read_h5ad(adata_dir_regressed)

    else:

        adata_list = []

        for donor in donor_keep_ct:

            adata_donor = adata_ct[adata_ct.obs.donor_id == donor]
            adata_donor.raw = adata_donor

            sc.pp.scale(adata_donor)

            sc.tl.score_genes_cell_cycle(adata_donor,
                                         s_genes=s_genes,
                                         g2m_genes=g2m_genes)

            adata_donor = adata_donor.raw.to_adata()

            # Regress approch 1
            obs    = adata_donor.obs.copy()
            design = pd.get_dummies(adata_donor.obs['sample_id'])
            design.columns = design.columns.to_list()
            design['S_score'] = adata_donor.obs['S_score'].values
            design['G2M_score'] = adata_donor.obs['G2M_score'].values
            adata_donor.obs = design # temporarily replace obs with design matrix

            sc.pp.regress_out(adata_donor, keys=design.columns.to_list(), n_jobs=n_jobs)
            adata_donor.obs = obs # reset obs

            # Regress approch 2
            # sc.pp.regress_out(adata_donor, keys=['sample_id'], n_jobs=n_jobs)
            # sc.pp.regress_out(adata_donor, keys=['S_score', 'G2M_score'], n_jobs=n_jobs)

            adata_list.append(adata_donor)

        adata_reg = anndata.concat(adata_list)
        adata_reg.write(adata_dir_regressed)

    # ----------------------
    # Identify HVGS per line
    # ----------------------

    var = variance(adata_reg)
    rank_var = np.argsort(var)[::-1][:num_hvgs]
    highly_variable = np.isin(np.arange(adata_reg.shape[-1]),rank_var)
    adata_reg.var['highly_variable'] = highly_variable

    hvgs_dict = {'all': adata_reg.var['highly_variable']}

    for donor in donor_keep_ct:

        logdir_line = os.path.join(logdir_exp,donor)
        os.makedirs(logdir_line, exist_ok=True)

        # Subset to query line
        adata_donor = adata_reg[adata_reg.obs.donor_id == donor]
        adata_donor.raw = adata_donor

        # --------------------------
        # Find highly variance genes
        # --------------------------
        var = variance(adata_donor)
        adata_donor.var['variance_line'] = var
        idx = np.argsort(var)[::-1][:num_hvgs_line]
        highly_variable_line = np.isin(np.arange(adata.shape[-1]),idx)
        adata_donor.var['highly_variable_line'] = highly_variable_line
        hvgs_dict[donor] = highly_variable_line

        # Subset
        idx = np.any([adata_donor.var['highly_variable'],
                      adata_donor.var['highly_variable_line']],0)
        adata_donor = adata_donor[:,idx]

        # Scale
        sc.pp.scale(adata_donor)

        # -------------
        # DR then plots
        # -------------
        sc.tl.pca(adata_donor, svd_solver='arpack')
        sc.pp.neighbors(adata_donor,n_neighbors=5,)
        sc.tl.umap(adata_donor)
        ## plots
        color = ['celltype', 'cluster_id', 'sample_index', 'pool_id']
        # umap
        sc.pl.umap(adata_donor, color=color)
        plt.savefig(os.path.join(logdir_line, 'umap.svg'))
        plt.close()
        # pca
        sc.pl.pca(adata_donor, color=color)
        plt.savefig(os.path.join(logdir_line, 'pca.svg'))
        plt.close()

        del adata_donor
        gc.collect()

    # --------------------------------------------------------
    # Take the union of all hvgs then subset for final anndata
    # --------------------------------------------------------

    idx_keep = np.any([v for k,v in hvgs_dict.items()],0)

    for donor in donor_keep_ct:

        logdir_line = os.path.join(logdir_exp,donor)

        # Subset to query line
        adata_donor = adata_reg[adata_reg.obs.donor_id == donor]
        adata_donor.raw = adata_donor
        adata_donor.var['highly_variable'] = idx_keep
        adata_donor = adata_donor[:,idx_keep]

        sc.pp.scale(adata_donor)

        # -------------
        # DR then plots
        # -------------
        sc.tl.pca(adata_donor, svd_solver='arpack')
        sc.pp.neighbors(adata_donor,n_neighbors=5,)
        sc.tl.umap(adata_donor)
        # plots
        color = ['celltype', 'cluster_id', 'sample_index', 'phase']
        sc.pl.umap(adata_donor, color=color)
        plt.savefig(os.path.join(logdir_line, 'umap_comb.svg'))
        plt.close()
        sc.pl.pca(adata_donor, color=color)
        plt.savefig(os.path.join(logdir_line, 'pca_comb.svg'))
        plt.close()

        # ----
        # Save
        # ----.
        adata_donor.write(os.path.join(logdir_line,'scaled_data.h5ad'))

        del adata_donor
        gc.collect()

    df_hvg = pd.DataFrame(hvgs_dict,index=adata.var_names.to_numpy())
    df_hvg.to_csv(os.path.join(logdir_exp,'hvg.csv'))

# ----------------------------
# Test for regression
# ----------------------------
## test
# adata_donor = adata[adata.obs.donor_id == donor]
# # sc.pp.regress_out(adata_donor,keys=['sample_id'],n_jobs=10)
# var = variance(adata_donor)
# rank_var = np.argsort(var)[::-1][:3000]
# highly_variable = np.isin(np.arange(adata_donor.shape[-1]),rank_var)
# adata_donor = adata_donor[:,highly_variable]
# sc.pp.scale(adata_donor)
# sc.tl.pca(adata_donor, svd_solver='arpack')
# sc.pp.neighbors(adata_donor,n_neighbors=5,)
# sc.tl.umap(adata_donor)
# # plots
# logdir_line = ''
# color = ['celltype', 'cluster_id', 'sample_index', 'pool_id']
# sc.pl.umap(adata_donor, color=color)
# plt.savefig(os.path.join(logdir_line, 'umap.svg'))
# plt.close()
# sc.pl.pca(adata_donor, color=color)
# plt.savefig(os.path.join(logdir_line, 'pca.svg'))
# plt.close()
#

# adata_donor = adata[adata.obs.donor_id == donor]
# adata_list = []
# for ct in ['P_FPP','FPP']:
#     adata_ct = adata_donor[adata_donor.obs.celltype == ct]
#     sc.pp.regress_out(adata_ct,keys=['sample_id'],n_jobs=10)
#     adata_list.append(adata_ct)
#
# adata_donor2 = anndata.concat(adata_list)
# var = variance(adata_donor2)
# rank_var = np.argsort(var)[::-1][:3000]
# highly_variable2 = np.isin(np.arange(adata_donor2.shape[-1]),rank_var)
# adata_donor2_hvg1 = adata_donor2[:,highly_variable]
# sc.pp.scale(adata_donor2_hvg1)
# sc.tl.pca(adata_donor2_hvg1, svd_solver='arpack')
# sc.pp.neighbors(adata_donor2_hvg1,n_neighbors=5,)
# sc.tl.umap(adata_donor2_hvg1)
# sc.pl.umap(adata_donor2_hvg1, color=color)
# plt.savefig(os.path.join(logdir_line, 'umap2-1.svg'))
# plt.close()
# sc.pl.pca(adata_donor2_hvg1, color=color)
# plt.savefig(os.path.join(logdir_line, 'pca2-1.svg'))
# plt.close()
# adata_donor2_hvg2 = adata_donor2[:,highly_variable2]
# sc.pp.scale(adata_donor2_hvg1)
# sc.tl.pca(adata_donor2_hvg2, svd_solver='arpack')
# sc.pp.neighbors(adata_donor2_hvg2,n_neighbors=5,)
# sc.tl.umap(adata_donor2_hvg2)
# sc.pl.umap(adata_donor2_hvg2, color=color)
# plt.savefig(os.path.join(logdir_line, 'umap2-3.svg'))
# plt.close()
# sc.pl.pca(adata_donor2_hvg1, color=color)
# plt.savefig(os.path.join(logdir_line, 'pca2-2.svg'))
# plt.close()
#
