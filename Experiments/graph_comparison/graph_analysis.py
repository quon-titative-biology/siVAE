import os,gc

import numpy as np
import pandas as pd

import rpy2

## Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

import siVAE
from siVAE.model.output import output_handler
from siVAE.util import reduce_dimensions


from sklearn.metrics import pairwise_distances

# stats
from scipy.stats import rankdata

def emb2adj(X, metric='euclidean', percentile=None, threshold=None):
    """
    Convert gene embeddings to adjacency based on similarity
    Input
        X: n_genes x n_dim array of gene embeddings
        metric: distance metric to use for adjacency
        percentile: percentage of
        threshold: specific distance metric for cutoff
    Return
        adjacency matrix (n_genes x n_genes)
    """
    #
    dist = pairwise_distances(X,metric=metric)
    #
    if threshold is None:
        upper = dist[np.triu_indices(len(dist),1)]
        threshold = np.percentile(upper,percentile*100)
    #
    adj = (dist < threshold)
    np.fill_diagonal(adj,False)
    return adj


def sim2adj(X, percentile=None, threshold=None):
    """
    Convert similarity matrix to adjacency matrix through thresholding
    For distance matrix, input X = -dist
    """
    if threshold is None:
        upper = X[np.triu_indices(len(X),1)]
        threshold = np.percentile(upper,(1-percentile)*100)
    adj = (X > threshold)
    np.fill_diagonal(adj,False)
    return adj


def adj2edge(X,node_names=None,threshold=0.1,directed=False):
    if not isinstance(X,pd.DataFrame):
        X = pd.DataFrame(X,columns=node_names,index=node_names)
    if not directed:
        idx_lower = np.tril_indices(X.shape[0])
        X.values[idx_lower] = False
    df_edges = X.stack().reset_index()
    df_edges.columns = ['node1','node2','weight']
    if threshold is not None:
        df_edges = df_edges[df_edges['weight'] > threshold]
    return df_edges

def min_intersect(Xs,threshold=1):
    """
    Input
        Xs: iterable of lists
        threshold: int or float, if int min number of appearance, if float between 0 and 1, min percentage of appearance
    Return
        item_valid: lists of items from input that appeared on lists above threshold
    """
    assert threshold > 0, f'Threshold value of {threshold} is not valid. Input positive value for threshold'
    if 1 <= threshold:
        threshold = int(threshold)
    else:
        threshold = int(len(Xs) * threshold)
    combined_items = np.concatenate(Xs)
    item,counts=np.unique(combined_items,return_counts=True)
    item_valid = item[counts >= threshold]
    return item_valid

def place_column_texts(ax, xy, texts, shift, n_wrap=10,**kwargs):
    n_col = int(np.ceil(len(texts)/n_wrap+1))
    x,y=xy
    for i in range(n_col):
        begin = i*n_wrap
        end = (i+1)*(n_wrap)
        text = texts[begin:end]
        text = "\n".join(text)
        ax.annotate(text,(x+shift*i, y), ha='left', va='top', **kwargs)
    return ax

def plot_texts(texts_list,ax=None,shift=0.35, n_wrap=11,**kwargs):
    """"""
    nrows=len(texts_list)
    ncols=len(texts_list[0])
    # set up ax
    if ax is None:
        _,ax = plt.subplots(figsize=(ncols*3,nrows*3))
    _ = ax.set_xlim([0, ncols])
    _ = ax.set_ylim([0, nrows])
    _ = ax.set_xticks(np.arange(0,ncols,1))
    _ = ax.set_yticks(np.arange(0,nrows,1))
    ax.grid(which='major',alpha=1)
    for axis in [ax.xaxis,ax.yaxis]:
        for tick in axis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
    # Plot texts
    for irow in range(nrows):
        for icol in range(ncols):
            xy = (icol + 0.1, nrows - 1 - irow + 0.9)
            texts = texts_list[irow][icol]
            ax = place_column_texts(ax,xy,texts,shift=shift,n_wrap=n_wrap,**kwargs)
    return ax

def map2colors(x,palette='RdBu',n_bin=5):
    series = pd.Series(x)
    if 'f' in x.dtype.str:
        series = pd.cut(series,n_bin).sort_values()
        series = series.astype('str')
    if isinstance(palette,str):
        palette = sns.color_palette(palette)
    lut = dict(zip(series.unique(),
                   palette))
    row_colors = series.map(lut)
    return row_colors


"""
Manually specify
    dataset: str, name
    logdir: str, dir where the siVAE results are stored
    downsampled: bool, whether the dataset was downsampled
"""

## FetalLiver
dataset = 'FetalLiver'
logdir = 'out/exp2/FetalLiver/liver-Kupffer_Cell'
downsampled=False

## AD dataset
dataset = 'AD_dataset'
logdir='out/AD_dataset/exp1'
downsampled=False

## iPSC cuomo
dataset = 'iPSC_cuomo'
logdir='out/exp2/iPSC_cuomo'
downsampled=False

## iPSC
dataset = 'iPSC'
logdir='out/iPSC_neuronal/D11/experiment/FPP'
logdir='out/exp1/iPSC_neuronal/data/iPSC_neuronal/D11/experiment/FPP'
logdir='out/exp2/iPSC_neuronal/P_FPP'
logdir='out/exp2/iPSC_neuronal/1000/P_FPP'
# logdir='out/exp3/iPSC_neuronal/1000/FPP'
# logdir='out/exp3/iPSC_neuronal/500/FPP'
# logdir='out/exp2/FetalLiver/liver-Kupffer_Cell'
downsampled=True
diff_only=False

# ==============================================================================
#               Load and process siVAE results for downstream analysis
# ==============================================================================

tasks = os.listdir(logdir)
tasks = [t for t in tasks if os.path.isdir(os.path.join(logdir,t)) and t != 'plot']
tasks = [t for t in tasks if t != 'HPSI0214i-poih_2-2']

"""
Load siVAE results and also save additional info
"""

embs   = []
sizes  = []
dcs    = []
exps   = []

for task in tasks:
    logdir_task = os.path.join(logdir,task)
    result = output_handler.load_pickle(os.path.join(logdir_task,'kfold-0/siVAE_result.pickle'))
    # Sample size
    sample_emb = result.get_sample_embeddings()
    size = sample_emb.shape[0]
    sizes.append(sample_emb.shape[0])
    # Gen embeddings
    gene_emb = result.get_feature_embeddings()
    embs.append(gene_emb)
    # Plot gene emb
    X,labels = reduce_dimensions(gene_emb)
    df_plot = pd.DataFrame(X,columns=labels)
    df_plot['Recon Loss'] = np.square(result.get_value('reconstruction')[0]-result.get_value('reconstruction')[1]).mean(0)
    g = sns.scatterplot(data = df_plot, x=labels[0],y=labels[1],hue='Recon Loss')
    plt.savefig(os.path.join(logdir_task,'gene_emb.svg'))
    plt.close()
    # Calculate and save degree centrality
    dc = 1-np.square(result.get_value('reconstruction')[0]-result.get_value('reconstruction')[1]).mean(0)
    dcs.append(dc)
    exps.append(result.get_value('reconstruction')[0])

# Set gene_names
if dataset == 'FetalLiver':
    gene_names = np.loadtxt('data/FetalLiver/gene_names.txt',dtype='str')
else:
    gene_names = result.get_value('var_names')

# If tasks were downsampled, remove the downsampled index
if downsampled:
    tasks = [t.rsplit('-',1)[0] for t in tasks]


"""
Load and set metadata
"""

def load_meta(dataset, tasks, **kwargs):
    if dataset == 'iPSC':
        # Efficiency
        df_eff = pd.read_csv('data/iPSC_neuronal/diff_efficiency_neur.csv')
        df_eff.index = df_eff.donor_id
        df_eff = df_eff.reindex(tasks)
        df_eff['task'] = tasks
        df_meta = df_eff[[c for c in df_eff.columns if 'efficiency' in c]]
        df_meta['efficiency_binned'] = pd.qcut(df_meta['diff_efficiency'], q=4, precision = 0).map(lambda x: x.right).astype('float')
        # logtask = 'data/iPSC_neuronal/D11/experiment/FPP'
        # metadatas = pd.read_csv(os.path.join(logtask,'lines-DA_efficiency.txt'))
    elif dataset == 'AD_dataset':
        meta_dict = {'State'   : [t[:2] for t in tasks],
                     'gender'  : [t.split('-')[-1] for t in tasks],
                     'combined': [t[:2]+"-"+t.split('-')[-1] for t in tasks]}
        df_meta = pd.DataFrame(meta_dict,index=tasks)
    elif dataset == 'FetalLiver':
        tasks = [t.split('_sub')[0] for t in tasks]
        meta_dict = {'Week' : [int(t.split('_',1)[0]) for t in tasks],
                     'F_id' : [t.split('-',1)[1].split('_',1)[0] for t in tasks]}
        df_meta = pd.DataFrame(meta_dict,index=tasks)
    elif dataset == 'iPSC_cuomo':
        df_meta = pd.DataFrame([t[:-1] for t in tasks],columns=['efficiency_binned'],index=tasks)
    # Add custom
    for k,v in kwargs.items():
        df_meta[k] = v
    return df_meta, df_eff

df_meta, df_eff = load_meta(dataset, tasks, size=sizes)

"""
Filter task based on size
"""

logdir_plot = os.path.join(logdir,'plot')

if dataset == 'iPSC':
    filters = []
    filters.append(np.array(sizes) > 0)
    if diff_only:
        filters.append(df_meta.diff_efficiency>0.2)
        logdir_plot = os.path.join(logdir_plot,'diff_only')
    else:
        logdir_plot = os.path.join(logdir_plot,'all')
    keep_idx = np.where(np.all(filters,axis=0))[0]
else:
    keep_idx = np.arange(len(embs))

os.makedirs(logdir_plot, exist_ok=True)

embs  = [embs[i] for i in keep_idx]
dcs   = [dcs[i] for i in keep_idx]
tasks = [tasks[i] for i in keep_idx]
sizes  = [sizes[i] for i in keep_idx]
df_meta = df_meta.iloc[keep_idx]
recons = [dc.mean() for dc in dcs]

sns.scatterplot(x=sizes,y=recons)
plt.savefig(os.path.join(logdir_plot,'size vs recon.svg'))
plt.close()

df_corr_eff_mean = pd.read_csv('iPSC_neuronal/corr_eff_meanexp.csv')
df_corr_eff_mean.index = df_corr_eff_mean['index']

X_plot = df_meta.diff_efficiency.sort_values().to_numpy().reshape(-1,1)
fig, ax = plt.subplots(figsize=(0.5,10))
sns.heatmap(X_plot[:47],
            cmap=sns.cubehelix_palette(as_cmap=True),
            linewidths=1,
            linecolor='black',
            ax=ax)
ax.tick_params(left=False, bottom=False)
plt.savefig(os.path.join(logdir_plot,'eff_heatmap.svg'))
plt.close()


# ==============================================================================
#   Perform downstream analysis (task visualization and HVCG identification)
# ==============================================================================

"""
Analysis
1. Identify top HVCGs and plot their correlations
1. Identify union of top n central genes to be used for downstream analysis
2. Identify genes with highest variable degree centrality
3. Perform graph kernel and visualize task x task similarity matrix
4. Plot agreement between central genes
"""

"""
Measure correlation between degree centrality and efficiency for top HVCGs
"""

from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

#### Measure HVCGs with DC as recon loss
#### Measure correlation between DC_siVAE and metadata
dcs_recon = np.array(dcs).transpose()
# dcs_recon = dcs_recon.argsort(axis=0)
corrmetric = spearmanr
corrs=np.array([[corrmetric(dcs_gene,df_meta[col]) for dcs_gene in dcs_recon] for col in df_meta.columns])
corr = corrs[:,:,0]
pval = corrs[:,:,1]
df_corr = pd.DataFrame(corr.transpose(),
                       columns=df_meta.columns,
                       index=gene_names)
df_pval = pd.DataFrame(pval.transpose(),
                       columns=df_meta.columns,
                       index=gene_names)

logdir_corr = os.path.join(logdir_plot,'corr')
os.makedirs(logdir_corr,exist_ok=True)
df_corr.to_csv(os.path.join(logdir_corr,'Corr-DCsiVAE_vs_meta_corr.csv'))
df_pval.to_csv(os.path.join(logdir_corr,'Corr-DCsiVAE_vs_meta_pval.csv'))

# Print stats
df_corr[df_corr.index.str.startswith('MT-')].median()
df_pval[df_corr.index.str.startswith('MT-')].median()

padj = multipletests(df_pval['diff_efficiency'],method='fdr_bh')[1]
df_padj = pd.DataFrame(padj,index=df_pval.index,columns=['padj'])
idx_sig = np.where(df_padj.padj < 0.05)[0]
df_padj_sig = df_pval[df_padj.padj < 0.05]

non_MT_genes = df_padj_sig[np.invert(df_padj_sig.index.str.startswith('MT-'))].index
MT_genes = df_padj_sig[(df_padj_sig.index.str.startswith('MT-'))].index

#### Correlation between efficiency and mean expression
padj = multipletests(df_corr_eff_mean['P-value'],method='fdr_bh')[1]
df_corr_eff_mean['P-adj'] = padj
df_corr_eff_mean.loc[MT_genes].median()
df_corr_eff_mean.loc[non_MT_genes].median()
df_corr_eff_mean.loc[non_MT_genes]['P-adj']

genes_sig_corr_mean = df_corr_eff_mean[df_corr_eff_mean['P-adj'] < 0.05].index
genes_sig_corr_mean = df_corr_eff_mean[df_corr_eff_mean['padj'] < 0.05].index
np.isin(non_MT_genes,genes_sig_corr_mean).sum() # number of genes with sig corr w/ DC and exp

#### Measure gene to gene correlation

def corrcoef(matrix,method='pearson'):
    """
    Vectorize correlation/p-value calculations for pearson and spearman
    """
    from scipy.special import betainc
    if method == 'spearman':
        from scipy.stats import rankdata
        matrix=rankdata(matrix,axis=1)
    elif method == 'pearson':
        pass
    else:
        raise Exception('Input valid method')
    r = np.corrcoef(matrix)
    rf = r[np.triu_indices(r.shape[0], 1)]
    df = matrix.shape[1] - 2
    ts = rf * rf * (df / (1 - rf * rf))
    pf = betainc(0.5 * df, 0.5, df / (df + ts))
    p = np.zeros(shape=r.shape)
    p[np.triu_indices(p.shape[0], 1)] = pf
    p[np.tril_indices(p.shape[0], -1)] = p.T[np.tril_indices(p.shape[0], -1)]
    p[np.diag_indices(p.shape[0])] = np.zeros(p.shape[0])
    return r, p

corrs = [corrcoef(exp[:,idx_sig].transpose(),method='spearman') for exp in exps]
corrs = np.array(corrs)
corrs = np.moveaxis(corrs,1,-1)

idx_mt = df_padj_sig.index.str.startswith('MT-')
idx_non_mt = np.invert(idx_mt)

# Correlations between MT genes
MT_corrs = corrs[:,idx_mt][:,:,idx_mt]
np.median(MT_corrs.mean(0)[:,:,0])
np.median(MT_corrs.mean(0)[:,:,1])
# Correlations between MT genes and non-MT genes
MT_non_MT_corrs = corrs[:,idx_mt][:,:,idx_non_mt]
np.median(MT_non_MT_corrs.mean(0)[:,:,0])
np.median(MT_non_MT_corrs.mean(0)[:,:,1])


## Show genes with significant correlations

def plot_corr(top_genes_dict, df_meta, gene_names, metric, logdir):
    """"""
    for setname,top_genes in top_genes_dict.items():
        idx = np.array([np.where(g==gene_names)[0] for g in top_genes]).reshape(-1)
        effs = df_meta.iloc[:,0]
        df_plot = pd.DataFrame(metric[idx].transpose(),columns=top_genes,index=effs)
        df_plot = df_plot.unstack().reset_index()
        df_plot.columns = ['Gene','Efficiency','Degree centrality']
        # Plot as line plot with regression line
        g = sns.lmplot(data=df_plot,x='Efficiency',y='Degree centrality',hue='Gene',ci=None)
        plt.savefig(os.path.join(logdir_corr,f'Eff vs DC_siVAE {setname}.svg'))
        plt.close()

if dataset == 'iPSC':
    top_genes_dict = {}
    # Set manually
    top_genes_dict['manual'] = ['MT-ND3','MT-ATP6','S100A11','MT-CYB','MT-CO1']
    # Set automatically
    corr = df_corr[df_pval.diff_efficiency < 0.05].abs()
    top_genes_dict['corr'] = corr.sort_values(df_corr.columns[0]).index[-30:]
    plot_corr(top_genes_dict, df_meta, gene_names, dcs_recon, logdir_corr)


"""
Use graph kernel to generate similarity matrix used for visualization
"""

#### Identifty union of top n central genes to be used for downstream analysis

def get_central_genes(dcs,n=100):
    """ """
    if not hasattr(n, '__iter__'):
        n = [n]
    if not hasattr(dcs, '__iter__'):
        dcs = [dcs]
    hcgs_dict = {'all': range(dcs[0].shape[0])}
    for n_genes in n:
        central_genes = [np.argsort(dc)[::-1][:n_genes] for dc in dcs]
        subset_genes = np.unique(np.array(central_genes))
        hcgs_dict[f'central-{n_genes}'] = subset_genes
    return hcgs_dict

hcgs_dict = get_central_genes(dcs,n=100)


#### Identify genes with highest variable degree centrality based on adj matrix
logdir_adj = os.path.join(logdir_plot,'adj')
os.makedirs(logdir_adj,exist_ok=True)

from scipy.stats import spearmanr

for setname,idx_gene in hcgs_dict.items():
    # Set up directory
    # idx_gene_corr_mean = np.where((df_corr_eff_mean['P-value'] < 0.05/3362))[0]
    # idx_gene = np.setdiff1d(idx_gene,idx_gene_corr_mean)
    logdir_p = os.path.join(logdir_plot,'sim',setname)
    os.makedirs(logdir_p, exist_ok=True)
    sim_dict = {}
    gene_names_ = gene_names[idx_gene]
    # Graph kernels
    for p in [0.01, 0.005, 0.001]:
        adjs = [emb2adj(gene_emb[idx_gene],percentile=p) for gene_emb in embs]
        #### Degree Centrality
        dcs_adjs = np.array([adj.sum(0) for adj in adjs]).transpose() # n_gene x n_cellline degree centrality
        corr=np.array([[spearmanr(dcs_gene,df_meta[col])[0] for dcs_gene in dcs_adjs] for col in df_meta.columns])
        df_corr = pd.DataFrame(corr.transpose(),
                               columns=df_meta.columns,
                               index=gene_names[idx_gene])
        df_corr = df_corr.fillna(0)
        df_corr = df_corr.abs().sort_values(df_corr.columns[0])
        df_corr.to_csv(os.path.join(logdir_adj,f"hvcg-{setname}-{p}.csv"))
        #### Plot correlation
        top_genes = np.array(['MT-ND3','MT-ATP6','S100A11','MT-CYB','MT-CO1'])
        top_genes = top_genes[np.isin(top_genes,gene_names_)]
        ####
        idx = np.concatenate([np.where(g==gene_names_)[0] for g in top_genes]).reshape(-1)
        effs = np.argsort(df_meta.iloc[:,0])
        df_plot = pd.DataFrame(dcs_adjs[idx].transpose(),columns=top_genes,index=effs)
        df_plot = df_plot.unstack().reset_index()
        df_plot.columns = ['Gene','Efficiency','Degree centrality']
        #
        g = sns.lmplot(data=df_plot,x='Efficiency',y='Degree centrality',hue='Gene',ci=None)
        plt.savefig(os.path.join(logdir_adj,f'Eff vs DC_adj_{p} {setname}.png'))
        plt.close()

"""
Use graph kernel to generate similarity matrix used for visualization
"""

#### Perform graph kernel and visualize similarity matrix
from grakel.graph import Graph
from grakel.kernels import ShortestPath, \
                           PyramidMatch, \
                           VertexHistogram, \
                           WeisfeilerLehman, \
                           RandomWalkLabeled

kernels_dict = {'ShortestPath': ShortestPath,
                'PyramidMatch': PyramidMatch,
                'VertexHistogram': VertexHistogram,
                'WeisfeilerLehman': WeisfeilerLehman,
                'RandomWalkLabeled': RandomWalkLabeled}

kernel_methods = ['VertexHistogram','WeisfeilerLehman']
# kernel_methods = ['VertexHistogram']


#### Save graph inferred GRN

#### On all genes GRN inference using rpy2
from rpy2.robjects import pandas2ri
from rpy2 import robjects

pandas2ri.activate() # use pandas dataframe as dataframe in R

robjects.r('''
           f <- function(exp,method='clr') {

                    library(minet)
                    adj.mat <- minet(exp,method)
            }
            ''')
minet = robjects.globalenv['f']

gene_info = pd.DataFrame(gene_names,columns=['Name'])

def map_type(g):
    if 'MT-' in g:
        t = 'MT'
    elif 'RPL' in g:
        t = 'RPL'
    elif 'RPS' in g:
        t = 'RPS'
    else:
        t = 'Else'
    return t

gene_info['Type'] = [map_type(g) for g in gene_names]

gene_info.to_csv(os.path.join(logdir_adj,'gene_info.csv'))
from scipy.stats import ranksums,mannwhitneyu

for setname,idx_gene in hcgs_dict.items():
    idx_gene = hcgs_dict[setname]
    # Set up directory
    if setname != 'all':
        logdir_p = os.path.join(logdir_plot,'sim',setname)
        os.makedirs(logdir_p, exist_ok=True)
        # Graph kernels
        for p in [0.01, 0.001]:
            logdir_adj_p = os.path.join(logdir_adj,setname,str(p))
            os.makedirs(logdir_adj_p,exist_ok=True)
            # Adjacency based on distance in gene embedding space or clr
            for adj_method in ['siVAE']:
                if adj_method == 'siVAE':
                    adjs = [emb2adj(gene_emb[idx_gene],percentile=p) for gene_emb in embs]
                elif adj_method == 'clr':
                    sims = [minet(exp[:,idx_gene],'clr') for exp in exps]
                    adjs = [sim2adj(sim,percentile=p) for sim in sims]
                for task,adj in zip(tasks,adjs):
                    eff = np.unique(df_eff.loc[task].diff_efficiency)[0]
                    gene_names_subset = gene_names[idx_gene]
                    df_edges = adj2edge(adj,gene_names_subset)
                    # Measure number of MTs
                    MT_count = [('MT-' in n1) + ('MT-' in n2) for n1,n2 in zip(df_edges.node1,df_edges.node2)]
                    df_edges['MT_count'] = MT_count
                    # Save
                    filename = os.path.join(logdir_adj_p,f"{int(eff*100)}-{task}-{adj_method}.csv")
                    print(filename)
                    print((df_edges.MT_count == 2).sum())
                    df_edges.to_csv(filename)
                # visualize edges
                MTidx = pd.Series(gene_names_subset).str.startswith('MT-')
                adj_mts = [adj[MTidx][:,MTidx] for adj in adjs]
                edges = [adj2edge(adj,
                                  node_names=gene_names_subset[MTidx],
                                  threshold=None,
                                  directed=False) for adj in adj_mts]
                df_edges = pd.concat([e['weight'] for e in edges],axis=1)
                df_edges.columns = tasks
                df_edges.index=edges[0].iloc[:,:2].agg("_".join,axis=1)
                df_edges = df_edges[df_edges.sum(1) > 0]
                eff = df_meta.drop_duplicates().reindex(df_edges.columns).diff_efficiency
                corrs = []
                for _,edge in df_edges.iterrows():
                    # corrs.append(spearmanr(edge,eff))
                    corrs.append(mannwhitneyu(eff[edge.to_numpy()],eff[np.invert(edge.to_numpy())]))
                    # corrs.append(ranksums(eff[edge.to_numpy()],eff[np.invert(edge.to_numpy())]))
                df_corr = pd.DataFrame(corrs)
                df_corr['correlation'] = df_corr.iloc[:,0]
                df_corr.pvalue[df_corr.correlation==0] = 1
                df_corr.correlation = df_corr.correlation.fillna(0).abs()
                df_corr.pvalue = df_corr.pvalue.fillna(1)
                df_corr['pvalue_corr'] = multipletests(df_corr.pvalue,method='fdr_bh')[1]
                df_corr['pvalue_bool'] = df_corr.pvalue_corr < 0.05
                df_corr.index = df_edges.index
                # Reindex
                index = df_corr.sort_values(['pvalue_bool','correlation']).index
                df_corr = df_corr.reindex(index)
                df_edges = df_edges.reindex(index)
                idx_col = eff.argsort().values
                df_edges = df_edges.iloc[:,idx_col]
                col_colors = [map2colors(eff[idx_col],n_bin=5,
                                         palette=sns.cubehelix_palette(as_cmap=False)).reset_index(drop=True)]
                row_colors = [map2colors(df_corr.correlation.abs(),'Blues'),
                              map2colors(df_corr.pvalue_bool,palette=['Red','Blue'])]
                # plot heatmap
                sns.clustermap(df_edges,
                               cmap='Blues',
                               col_cluster=False,
                               row_cluster=False,
                               col_colors=col_colors,
                               row_colors=row_colors)
                plt.savefig(os.path.join(logdir_adj_p,'edges.svg'))
                plt.close()
                df_corr.to_csv(os.path.join(logdir_adj_p,'sig_edges.csv'))

#### For adjacency matrix, examine change in connections

edge_types = ['MT_inner','MT_outer','RP_inner','RP_outer'] # edge types to examine
adj_methods = ['siVAE'] # Method to infer adj
percentiles = [0.01] # Percentile of adj

for setname,idx_gene in hcgs_dict.items():
    if setname != 'all':
        logdir_p = os.path.join(logdir_plot,'adj',setname)
        os.makedirs(logdir_p, exist_ok=True)
        sim_dict = {}
        # Graph kernels
        nrows = len(percentiles)
        ncols = len(edge_types)
        for p in percentiles:
            logdir_adj_p = os.path.join(logdir_adj,setname,str(p))
            os.makedirs(logdir_adj_p,exist_ok=True)
            df_stats = []
            for adj_method in adj_methods:
                print(adj_method)
                stat = []
                # Adjacency based on distance in gene embedding space or clr
                if adj_method == 'siVAE':
                    adjs = [emb2adj(gene_emb[idx_gene],percentile=p) for gene_emb in embs]
                elif adj_method == 'clr':
                    sims = [minet(exp[:,idx_gene],'clr') for exp in exps]
                    adjs = [sim2adj(sim,percentile=p) for sim in sims]
                # Subset
                gene_names_subset = gene_names[idx_gene]
                MT_idx = pd.Series(gene_names_subset).str.startswith('MT-')
                RP_idx = gene_info[(gene_info.Type == 'RPL') | (gene_info.Type == 'RPS')].Name
                RP_idx = np.isin(gene_names_subset,RP_idx)
                # GRN per task
                for task,adj in zip(tasks,adjs):
                    eff = np.unique(df_eff.loc[task].diff_efficiency)[0]
                    #
                    MT_inner = np.sum(adj[MT_idx][:,MT_idx])
                    MT_outer = np.sum(np.delete(adj,MT_idx,axis=0)[:,MT_idx])
                    RP_inner = np.sum(adj[RP_idx][:,RP_idx])
                    RP_outer = np.sum(np.delete(adj,RP_idx,axis=0)[:,RP_idx])
                    stat.append((eff,MT_inner,MT_outer,RP_inner,RP_outer))
                # df_stat
                df_stat = pd.DataFrame(stat,columns=['eff']+edge_types)
                df_stat['method'] = adj_method
                df_stats.append(df_stat)
            df_stats = pd.concat(df_stats).reset_index(drop=True)
            # Plot lines through
            df_plot = df_stats.melt(['eff','method'])
            sns.lmplot(data=df_plot,x='eff',y='value',ci=0,hue='method',col='variable',sharey=False)
            plt.savefig(os.path.join(logdir_adj_p,'lineplots.svg'))
            plt.close()
        spearmanr(df_stat.eff,df_stat.MT_inner)
        spearmanr(df_stat.eff,df_stat.MT_outer)
        spearmanr(df_stat.eff,df_stat.RP_inner)
        spearmanr(df_stat.eff,df_stat.RP_outer)


for setname,idx_gene in hcgs_dict.items():
    # Set up directory
    idx_gene_corr_mean = np.where((df_corr_eff_mean['P-value'] < 0.05/3362))[0]
    idx_gene = np.setdiff1d(idx_gene,idx_gene_corr_mean)
    logdir_p = os.path.join(logdir_plot,'sim',setname)
    os.makedirs(logdir_p, exist_ok=True)
    sim_dict = {}
    # Graph kernels
    for p in [0.01, 0.001]:
        adjs = [emb2adj(gene_emb[idx_gene],percentile=p) for gene_emb in embs]
        graphs = [Graph(adj,{i:v for i,v in enumerate(adj.sum(0))},{},'adjacency') for adj in adjs]
        for method in kernel_methods:
            gk = kernels_dict[method](normalize=True,n_jobs=3)
            sim = gk.fit_transform(graphs)
            sim_dict[f'{method}-{p}'] = sim
            gc.collect()
    # Frobenius norm
    # def frobnorm(emb1,emb2):
    #     from sklearn.metrics import pairwise_distances
    #     dist = pairwise_distances(emb1,emb2)
    #     norm = np.linalg.norm(dist)
    #     return norm
    # sim = [[frobnorm(emb1[idx_gene],emb2[idx_gene]) for emb1 in embs] for emb2 in embs]
    # sim_dict['Frob'] = np.array(sim)
    #
    # Iterate through sim_dict and plot
    for method,sim in sim_dict.items():
        #
        for red_method in ['PCA','tSNE','UMAP']:
            red_kwargs = {}
            if red_method == 'tSNE':
                red_kwargs['perplexity'] = 3
            elif red_method == 'UMAP':
                red_kwargs['min_dist'] = 0.0
                red_kwargs['n_neighbors'] = 4
            # Create dataframe
            X,labels = reduce_dimensions(sim,method=red_method,**red_kwargs)
            df_plot = pd.DataFrame(X,columns=labels)
            df_plot['tasks'] = tasks
            df_plot = df_plot.groupby('tasks').mean()
            df_meta_ = df_meta
            # df_meta_ = df_meta.drop_duplicates().loc(df_plot.index)
            df_meta_ = df_meta_.drop_duplicates().loc[df_plot.index]
            #
            # df_meta_['Efficiency rank'] = np.argsort(df_meta_['diff_efficiency'])
            df_plot = pd.concat([df_plot,df_meta_],axis=1)
            #
            # Plot
            for col in df_meta.columns.to_list():
                ax = sns.scatterplot(data = df_plot,
                                x=labels[0],
                                y=labels[1],
                                hue=col,
                                s=150,
                                linewidth=0.5,
                                edgecolor='black'
                                # palette=sns.color_palette("vlag", as_cmap=True)
                                )
                ax.legend([],[], frameon=False)
                plt.tight_layout()
                plt.savefig(os.path.join(logdir_p,f'{method}-similarity_{red_method}-{col}.svg'))
                plt.close()
        #
        df_plot = pd.DataFrame(sim,columns=tasks,index=tasks)
        sns.clustermap(df_plot)
        plt.savefig(os.path.join(logdir_p,f'{method}-similarity_heatmap.svg'))
        plt.close()
        gc.collect()


"""
Perform GSEA on the correlation coefficient with efficiency
"""


from siVAE import gsea
import gseapy as gp

logdir_gsea = os.path.join(logdir_plot,'gsea')
os.makedirs(logdir_gsea,exist_ok=True)

df_rnk = df_corr['diff_efficiency'].abs()
# for gsname in ['KEGG','Hallmark','scsig','MT']:
for gsname in ['MT']:
    logdir_gs = os.path.join(logdir_gsea,gsname)
    os.makedirs(logdir_gs,exist_ok=True)
    if gsname=='MT':
        gs_dict = gsea.load_geneset('KEGG')
        # gs_dict.update(gsea.load_geneset('Hallmark'))
        gs_dict['MT'] = df_rnk[df_rnk.index.str.startswith('MT-')].index.to_list()
    else:
        gs_dict = gsea.load_geneset(gsname)
    pre_res = gp.prerank(rnk=df_rnk,
                         gene_sets=gs_dict,
                         outdir=logdir_gs,
                         min_size=1,
                         permutation_num=10000)
    df_res = pre_res.res2d
    df_res['nlogpval'] = -np.log(df_res.fdr+1e-5)
    df_res = df_res.sort_values(['nlogpval'],ascending=False)
    df_res['Geneset'] = df_res.index
    df_res['sig'] = df_res.fdr < 0.05
    plt.close()
    sns.barplot(data=df_res.iloc[:10],x='nlogpval',y='Geneset',hue='sig',dodge=False)
    plt.tight_layout()
    plt.savefig(os.path.join(logdir_gs,'GSEA_pval.svg'))
    plt.close()

gs_genes = np.unique(sum(gs_dict.values(),[]))

df_rnk = pd.DataFrame(df_rnk.sort_values(ascending=False))
df_rnk['MT'] = 'Non-MT'
for complex in ['CO','ND','ATP','CYB']:
    df_rnk.loc[df_rnk.index.str.startswith('MT-'+complex),'MT'] = complex

df_plot = df_rnk.copy()
df_plot = df_plot[np.isin(df_plot.index,gs_genes)]
df_plot['Gene'] = df_plot.index
palette = {'Non-MT':'#FFFFFF',
           'ATP':'#66C2A5',
           'CO':'#FC8D62',
           'CYB':'#8DA0CB',
           'ND':'#E78AC3'}
g = sns.barplot(data=df_plot.iloc[:50],
                x='Gene',
                y='diff_efficiency',
                hue='MT',
                dodge=False,
                edgecolor ='black',
                linewidth=1,
                palette=palette
                )
# g.legend([],[], frameon=False)
plt.savefig(os.path.join(logdir_gsea,'GSEA_ranked_top_100.svg'))
plt.close()
