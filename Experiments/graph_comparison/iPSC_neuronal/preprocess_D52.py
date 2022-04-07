import os, gc
import scanpy as sc
import anndata

import numpy as np
import pandas as pd

## Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

adata1 = anndata.read_h5ad('D30.h5')
adata2 = anndata.read_h5ad('D52.h5')
adata  = anndata.concat([adata1,adata2])

logdir = 'data/D30-D52'
os.makedirs(logdir, exist_ok=True)

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


# ==============================================================================
#                             Calculate new efficiency
# ==============================================================================

# -----------------------------------
# Import differential efficiency data
# -----------------------------------

df_eff = pd.read_csv('diff_efficiency.csv')
df_eff.index = df_eff.donor_id
donor_overlap = np.intersect1d(df_eff.donor_id,adata.obs.donor_id)

df_eff = df_eff.sort_values('diff_efficiency')

## Remove lines not in differential efficiency data or
donor_keep = df_eff[df_eff.in_study != 'not_assessed'].donor_id.to_numpy()
adata = adata[np.isin(adata.obs.donor_id,donor_keep)]

line_counts = adata.obs.donor_id.value_counts()
df_plot = pd.DataFrame({"donor_id"   : line_counts.index,
                        'Counts'     : line_counts.values,
                        'Efficiency' : df_eff.reindex(line_counts.index).diff_efficiency.values})

df_eff = df_eff[np.isin(df_eff.donor_id,donor_keep)]

# df_lines_keep = df_plot[(df_plot.Counts > 1000) & (df_plot.Counts < 8000)]
# donor_keep = df_lines_keep.donor_id.values

##
adata_D52 = adata[adata.obs.time_point == 'D52']
celltype = adata_D52.obs.celltype.copy()
celltype[np.invert(np.isin(celltype,['DA','Sert']))] = 'NonNeuronal'
adata_D52.obs['celltype_neur'] = celltype
value_counts = adata_D52.obs.value_counts(['donor_id','celltype_neur'])
value_counts = value_counts.unstack().fillna(0)
values_percent = value_counts.div(value_counts.sum(1),axis=0)

df_plot = values_percent.reset_index().melt(id_vars='donor_id',value_name='efficiency')
df_plot = df_plot[df_plot.celltype_neur != 'NonNeuronal']

sns.barplot(data = df_plot,
            x='donor_id',
            y='efficiency',
            hue='celltype_neur')
plt.savefig('eff_new.svg')
plt.close()

values_percent['diff'] = (values_percent['DA'] - values_percent['Sert']).abs()
sns.barplot(data = values_percent.reset_index(),
            x='donor_id',
            y='diff')
plt.savefig('eff_new_diff.svg')
plt.close()

values_percent = values_percent.reindex(df_eff.donor_id)
df_eff['DA_efficiency'] = values_percent.DA
df_eff['Sert_efficiency'] = values_percent.Sert
df_eff.to_csv('diff_efficiency_neur.csv')
delta_eff = df_eff.diff_efficiency - (df_eff.DA_efficiency + df_eff.Sert_efficiency)

# ==============================================================================
#                                 Pre-processing
# ==============================================================================

# Violin plots
# ------------

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

# -----------------------------------
# Subset adata with selected donor_id
# -----------------------------------

adata = adata[np.isin(adata.obs.donor_id,donor_keep)]

value_counts = adata.obs['donor_id'].value_counts()
value_counts = value_counts[value_counts != 0]
adata.obs['donor_id'] = pd.Categorical(adata.obs['donor_id'],
                                       value_counts.index.astype('str'))

neur_ct  = ['Sert', 'DA', 'Astro', 'Epen1', 'U_Neur1', 'U_Neur3', 'Epen2', 'P_Sert']
neur2_ct = ['Sert', 'DA']


for treatment in ['all','NONE','ROT']:
    #
    if treatment != 'all':
        obs1 = adata.obs[adata.obs['treatment'] == treatment]
    else:
        obs1 = adata.obs
    #
    # Cell groups
    # for ct in ['all','neur','neur2']:
    #     #
    #     if ct == 'neur':
    #         obs2 = obs1[np.isin(obs1.celltype,neur_ct)]
    #     elif ct == 'neur2':
    #         obs2 = obs1[np.isin(obs1.celltype,neur2_ct)]
    #     else:
    #         obs2 = obs1
    #     #
    #     ax = sns.histplot(obs2,
    #                       x = 'donor_id',
    #                       hue = 'celltype',
    #                       multiple='stack')
    #     plt.savefig(os.path.join(logdir,'Lines_count_by_ct-{}={}-{}={}.svg'.format('treatment',
    #                                                                                treatment,
    #                                                                                'cellgroups',
    #                                                                                ct)))
    #     plt.close()
    # Individual cell types
    celltypes = ['P-FPP',"FPP",'DA','Sert']
    fig,axes = plt.subplots(len(celltypes),sharex=False, sharey= False, figsize = (10,15))
    for ax,ct in zip(axes,celltypes):
        #
        obs2 = obs1[np.isin(obs1.celltype,[ct])]
        value_counts_ct = obs2.donor_id.value_counts()
        value_counts_ct = value_counts_ct[value_counts_ct > 200]
        obs2 = obs2[np.isin(obs2.donor_id,value_counts_ct.index)]
        #
        #
        ax = sns.barplot(x=np.arange(len(value_counts_ct.index)),
                        y=value_counts_ct.values,
                        ax=ax)
        _ = ax.set_title(ct)
        _ = ax.set_xlabel('Cell line')
        _ = ax.set_ylabel('Counts')
        #
    plt.tight_layout()
    plt.savefig(os.path.join(logdir,'Lines_count_by_ct-{}={}-{}.svg'.format('treatment',
                                                                            treatment,
                                                                            'celltypes')))
    plt.close()

celltypes = ['DA','Sert']
treatments = ['NONE','ROT']

for treatment in treatments:
    adata_tr = adata[adata.obs['treatment'] == treatment]
    for ct in celltypes:
        adata_ct = adata_tr[np.isin(adata_tr.obs.celltype,[ct])]
        #
        value_counts_ct = adata_ct.obs.donor_id.value_counts()
        value_counts_ct = value_counts_ct[value_counts_ct > 200]
        donor_keep = value_counts_ct.index.to_numpy()
        adata_ct = adata_ct[np.isin(adata_ct.obs.donor_id,donor_keep)]
        #
        var = variance(adata_ct)
        for donor in adata_ct.obs.donor_id.unique():
            adata_donor = adata_ct[np.isin]
