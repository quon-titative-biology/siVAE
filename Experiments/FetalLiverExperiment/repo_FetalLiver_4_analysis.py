#### Analysis on experiments
"""
This file performs downstream analysis on the output of the siVAE/VAE.

0. Basic analysis on the siVAE results/ Input dataset

1. siVAE loadings comparison
    a.

2. Gene embeddings
    a. Geneset analysis: create scatterplot of embeddings annotated with genesets
    b. Identification of central genes
    c. Identification of neigborhood genes

"""


# Built-in
import os
import gc

# Libs
import numpy as np
import pandas as pd
from scipy import stats

# Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap

## siVAE
from siVAE import util
from siVAE.model.output import analysis
from siVAE.util import reduce_dimensions
from siVAE.util import remove_spines
from siVAE.model.output.output_handler import get_feature_attributions

#### ===========================================================================
#### Inputs
#### ===========================================================================

#### Set direcotires 0206 -----------------------------------------------------------
logdir_exp = 'out'

## Settings
do_Loadings = True
do_GeneEmbedding   = True
do_GenesetAnalysis = True
do_CentralGenes    = True
do_NeighborGenes   = True

#### Directories for models
siVAE_models = ['siVAE','siVAE-0','siVAE-linear-NB','siVAE-NB']
siVAE_models_dict = {name: os.path.join(logdir_exp, name, 'kfold-0') for name in siVAE_models}

scVI_models = ['scVI', 'LDVAE']
scVI_models_dict = {name: os.path.join(logdir_exp, name) for name in scVI_models}

logdir_GR  = os.path.join('out/siVAE','gene_relevance')
logdir_GRN = os.path.join('out','GRN')

logdir_dict = dict(siVAE_models_dict)
logdir_dict.update(scVI_models_dict)
logdir_dict['gene_relevance'] = logdir_GR
logdir_dict['GRN'] = logdir_GRN

#### Load ----------------------------------------------------------------------
logdir_analysis = os.path.join(logdir_exp,'analysis')
os.makedirs(logdir_analysis,exist_ok=True)

## Save the train/test split for consistency
datadir = os.path.join('out/data_dict.pickle')
data_dict = util.load_pickle(datadir)
datah_sample = data_dict['sample']
plot_args = data_dict['plot_args']

## Load DegreeCentralityPrediction
logdir_single = 'out/DegreeCentralityPrediction'
npz        = np.load(os.path.join(logdir_single,'single_gene_exp.npz'),allow_pickle=True)
PVEs       = npz['PVE']
recon_loss = npz['recon_loss'].mean(1)
Gene       = npz['gene_names']
Gene = np.load(os.path.join(logdir_single,'gene_name.npy'),allow_pickle=True)
df_single = pd.DataFrame({'Gene'      : Gene,
                          'Recon Loss': recon_loss,
                          'PVE'       : PVEs})
df_single.index = df_single['Gene']


#### ===========================================================================
#### 0. Basic analysis on the siVAE results/ Input dataset
#### ===========================================================================

## Set logdir
logdir_stats = os.path.join(logdir_analysis,'stats')
os.makedirs(logdir_stats,exist_ok=True)

## Load siVAE result
siVAE_result_dir = os.path.join(logdir_dict['siVAE'],"siVAE_result.pickle")
siVAE_result = util.load_pickle(siVAE_result_dir)

## Extract reconstruction loss per gene by averaging acorss cells
result = siVAE_result.get_model()
y      = result.get_value('reconstruction')[0]
y_pred = result.get_value('reconstruction')[1]
recon_loss_per_gene = np.square(y - y_pred).mean(0)

## Plot recon
gene_names = siVAE_result.get_value('feature').get_value('labels')
dropouts = datah_sample.X.var['pct_dropout_by_counts'].loc[gene_names]
df_plot = pd.DataFrame({'Percent dropout': dropouts,
                        'Recon loss'     : recon_loss_per_gene})

ax = sns.scatterplot(data=df_plot,x='Recon loss',y='Percent dropout',s=30,edgecolor='black')
remove_spines(ax,show_legend=False)
plt.savefig(os.path.join(logdir_stats,'Recon loss vs percent dropout.svg'))
plt.close()

#### ===========================================================================
#### 1. siVAE loadings comparison
#### ===========================================================================

## if do_loadings:

## Set logdir
logdir_loadings = os.path.join(logdir_analysis,'Loadings')
os.makedirs(logdir_loadings,exist_ok=True)

def extract_loadings(siVAE_result, do_siVAE=True, do_FA=True, method_DE=None, method='',
                     method_name=[], method_type=[], loadings=None, infer_FA_method=1):
    """
    do_FA: extract loadings inferred from feature attribution
    """
    ##
    loadings_list = []
    if loadings is not None: loadings_list += [loadings]
    ## Extract siVAE Loadings
    if do_siVAE:
        #### need chnge
        # v_mu = siVAE_result.get_feature_embeddings().transpose()
        v_mu = siVAE_result.get_gene_embeddings().transpose()
        v_mu = np.expand_dims(v_mu,0)
        loadings_list += [v_mu]
        method_name.append(method)
        method_type += [method]
    ## Extract Loadings from Feature Attribution
    if do_FA:
        scores, method_DE = get_feature_attributions(siVAE_result)
        scores = scores['decoder']
        FA_loadings = analysis.infer_FA_loadings(np.swapaxes(scores,1,2),
                                                       method=infer_FA_method)
        FA_loadings = FA_loadings[::-1]
        method_DE = method_DE[::-1]
        # Update methods list
        loadings_list += [FA_loadings]
        method_name += ["{}-{}".format(method,m) for m in method_DE]
        method_type += [method] * len(method_DE)
    loadings = np.concatenate(loadings_list,axis=0)
    return loadings, method_name, method_type

#### Load loadings -------------------------------------------------------------


#### siVAE feature embeddings + feature attributions
methods_loadings = []
method_type = []
infer_FA_method=1
all_loadings, method_loadings, method_type = extract_loadings(siVAE_result,
                                                          method='siVAE',
                                                          method_name=methods_loadings,
                                                          method_type=method_type)

#### Gene relevance input
for k in [10,100,200,1000,'default']:
    logdir_K = os.path.join(logdir_dict['gene_relevance'],'K-{}'.format(k))
    npz = np.load(os.path.join(logdir_K,'gene_relevance_result.npz'),allow_pickle=True)
    gene_names = npz['genes']
    gene_relevance = np.expand_dims(npz['partials'],0)
    gene_relevance = np.moveaxis(gene_relevance,3,1)
    gene_relevance = np.nan_to_num(gene_relevance,0)
    gr_loadings = analysis.infer_FA_loadings(gene_relevance,
                                             method=infer_FA_method)
    all_loadings = np.concatenate([all_loadings,gr_loadings],axis=0)
    methods_loadings += ['Gene Relevance (K={})'.format(k)]
    method_type  += ['Gene Relevance']

# #### Feature Attribution VAE
# siVAE_result_dir_VAE = os.path.join(logdir_VAE,"siVAE_result.pickle")
# siVAE_result_VAE = util.load_pickle(siVAE_result_dir_VAE)
# all_loadings, methods_loadings, method_type = extract_loadings(siVAE_result_VAE,
#                                                           do_siVAE=False,
#                                                           method='scVI',
#                                                           method_name=methods_loadings,
#                                                           method_type=method_type,
#                                                           loadings=all_loadings)

#### Feature Attribution siVAE_gamma=0
logdir_siVAE0 = logdir_dict['siVAE-0']
siVAE_result_dir_siVAE = os.path.join(logdir_siVAE0,"siVAE_result.pickle")
siVAE_result_siVAE = util.load_pickle(siVAE_result_dir_siVAE)
all_loadings, methods_loadings, method_type = extract_loadings(siVAE_result_siVAE,
                                                          method='siVAE0',
                                                          method_name=methods_loadings,
                                                          method_type=method_type,
                                                          loadings=all_loadings)

#### Plots ---------------------------------------------------------------------

#### Perform DR on latent embeddings and plot as annotated embeddings
from sklearn.preprocessing import scale
for LE_dim,loadings_in in enumerate(np.swapaxes(all_loadings,0,1)):
    for do_scale in [True,False]:
        if do_scale:
            loadings = scale(loadings_in,axis=1)
        else:
            loadings = loadings_in
        for dr_method in ['PCA','UMAP']:
            X_plot, dim_labels = reduce_dimensions(loadings, reduced_dimension = 2,
                                                   tsne_min=5, method = dr_method)
            df_plot = pd.concat([pd.DataFrame(X_plot), pd.DataFrame(methods_loadings)], axis = 1)
            df_plot.columns = dim_labels + ["Method"]
            df_plot['Method Type'] = method_type
            # Plot
            ax = sns.scatterplot(x = dim_labels[0], y = dim_labels[1],
                                 hue = "Method Type", data = df_plot, s=300)
            # add annotations one by one with a loop
            for line in range(0,df_plot.shape[0]):
                 _ = ax.text(df_plot[dim_labels[0]][line], df_plot[dim_labels[1]][line], df_plot['Method'][line],
                         horizontalalignment='left', size=5, color='black', weight='semibold')
            remove_spines(ax)
            plt.savefig(os.path.join(logdir_loadings,'loadings_LE_dim_scatterplot-scale_{}-{}-{}.svg'.format(do_scale,dr_method,LE_dim+1)))
            plt.close()

#### Plot loadings with reconstruction loss as hue for all methods
# Extract reconstruction loss per gene by averaging acorss cells
result = siVAE_result.get_model()
y      = result.get_value('reconstruction')[0]
y_pred = result.get_value('reconstruction')[1]
recon_loss_per_gene = np.square(y - y_pred).mean(0)

for FA_name,FA_loading in zip(methods_loadings, all_loadings):
    FA_loading = FA_loading.transpose()
    X_plot, dim_labels = reduce_dimensions(FA_loading, reduced_dimension = 2, method = 'tSNE')
    df_plot = pd.concat([pd.DataFrame(X_plot), pd.DataFrame(gene_names)], axis = 1)
    df_plot.columns = dim_labels + ["Label"]
    df_plot['Recon Accuracy'] = 1-recon_loss_per_gene
    # Plot
    ax = sns.scatterplot(x = dim_labels[0], y = dim_labels[1], hue = "Recon Accuracy", data = df_plot)
    remove_spines(ax)
    plt.title('Reconstruction Accuracy')
    plt.savefig(os.path.join(logdir_loadings,'loadings_scatterplot-{}.pdf'.format(FA_name)))
    plt.close()

# #### Plot loading vs loading for paired up methods
# for FA_name,FA_loading in zip(methods_loadings[1:],all_loadings[1:]):
#     analysis.compare_loadings2(values_dict,logdir_loadings,
#                                loadings=FA_loading,loadings_name=FA_name,
#                                hue=recon_loss_per_gene)

#### Pair-wise correlation matrix between loadings
from scipy import stats

def spearmanr(x,y,**kwargs):
    rho,pval = stats.spearmanr(x,y,**kwargs)
    n = len(x)
    t = rho * np.sqrt((n-2)/(1-np.square(rho)))
    prob = stats.t.sf(np.abs(t),n-2)*2


def max_spearmanr(matrix,axis=1,neg=False):
    if axis == 0:
        matrix = matrix.transpose()
    if neg:
        matrix2 = -matrix
    else:
        matrix2 = matrix
    results = [[stats.spearmanr(m1,m2) for m2 in matrix2] for m1 in matrix]
    corr   = np.array([[r[0] for r in rr] for rr in results])
    pvalue = np.array([[r[1] for r in rr] for rr in results])
    return corr,pvalue

# Libraries
from scipy.cluster import hierarchy

## Calculation pair-wise correlation matrix per latent dimension
loadings_ = np.swapaxes(all_loadings,0,1) # LE_dim x methods  x feature
corr_list = [np.abs(max_spearmanr(mat_))[0] for mat_ in loadings_] # correleation matrix per latent dimension
corr_list.append(np.array(corr_list).mean(0)) # append the average of the correlations

## Print out statistics
coefs = np.array([np.abs(max_spearmanr(mat_))[0] for mat_ in loadings_])
pvals = np.array([np.abs(max_spearmanr(mat_))[1] for mat_ in loadings_])

#### Need to set up manually
# idx_dict = {'siVAE': [0],
#             'DeepLIFT': [1],
#             'Feature Attribution': [1,2,3],
#             'Gene Relevance': [4,5,6,7,8]}
#
# comparisons = []
# comparisons.append(['siVAE','DeepLIFT'])
# comparisons.append(['siVAE','Feature Attribution'])
# comparisons.append(['siVAE','Gene Relevance'])
# comparisons.append(['Gene Relevance','Gene Relevance'])
# comparisons.append(['Gene Relevance','Feature Attribution'])
# comparisons.append(['siVAE','siVAE'])
# comparisons.append(['Feature Attribution','Feature Attribution'])
#
# stats_comparisons = []
# for t1,t2 in comparisons:
#     stats_list = []
#     for mat_ in [coefs,pvals]:
#         mat_ = mat_[np.array(idx_dict[t1])][:,np.array(idx_dict[t2])]
#         if t1==t2:
#             mat_ = np.triu(mat_).reshape(-1)
#             mat_ = mat_[mat_ != 0]
#         else:
#             mat_ = mat_.reshape(-1)
#         stats_list.append(mat_.mean())
#         stats_list.append(np.median(mat_))
#     stats_comparisons.append(stats_list)
#
#
# stats_comparisons = []
# for t1,t2 in comparisons:
#     stats_list = []
#     for mat_ in [coefs,pvals]:
#         mat_ = mat_[:,np.array(idx_dict[t1])][:,:,np.array(idx_dict[t2])]
#         if t1==t2:
#             mat_ = np.triu(mat_).reshape(-1)
#             mat_ = mat_[mat_ != 0]
#         else:
#             mat_ = mat_.reshape(-1)
#         stats_list.append(mat_.mean())
#         stats_list.append(np.median(mat_))
#     stats_comparisons.append(stats_list)

# comparisons_str = ["{} vs {}".format(c1,c2) for c1,c2 in comparisons]
# df_stats = pd.DataFrame(stats_comparisons,
#                         columns = ['Coef Mean','Coef Median', 'Pval Mean', 'Pval Median'],
#                         index = comparisons_str)

## Map colors for method grouping
colorlist = sns.color_palette("tab10")
# colorlist = ['blue','orange','red','green']
method2color = {t:c for t,c in zip(np.unique(method_type),colorlist)}
method2color['siVAE'] = colorlist[0]
method2color['Gene Relevance'] = colorlist[1]
method2color['scVI'] = colorlist[2]
method2color['siVAE0'] = colorlist[3]

for ii,corr in enumerate(corr_list):
    ## Create data frame for correlation matrix
    df = pd.DataFrame(np.abs(corr),
                      index   = methods_loadings,
                      columns = methods_loadings)
    colors = pd.Series(method_type).map(method2color).to_numpy()
    ## Clustermap
    ax=sns.clustermap(data=df,cmap='Blues',vmin=0,vmax=1,
                      row_colors=colors, col_colors=colors,square=True,
                      row_cluster=False,col_cluster=False)
    plt.savefig(os.path.join(logdir_loadings,
                             'loadings_corr-{}.svg'.format(ii+1)))
    plt.close()
    ## Dendrogram only
    Z = hierarchy.linkage(df, 'ward')
    _ = hierarchy.dendrogram(Z, leaf_rotation=90, leaf_font_size=8, labels=df.index)
    plt.tight_layout()
    plt.savefig(os.path.join(logdir_loadings,
                             'loadings_corr_dendrogram-{}.svg'.format(ii+1)))
    plt.tight_layout()
    plt.close()
    ## Clustermap with dendrogram
    ax=sns.clustermap(data=df,cmap='Blues',vmin=0,vmax=1,
                      row_linkage=Z, col_linkage=Z,square=True)
    plt.tight_layout()
    plt.savefig(os.path.join(logdir_loadings,
                             'loadings_corr2-{}.svg'.format(ii+1)))
    plt.close()


## Plot for subset
subset_idx = np.arange(9)
for ii,corr in enumerate(corr_list):
    ## Create data frame for correlation matrix
    corr = corr[subset_idx][:,subset_idx]
    methods_loadings_ = [methods_loadings[ii] for ii in subset_idx]
    method_type_ = [method_type[ii] for ii in subset_idx]
    df = pd.DataFrame(np.abs(corr),
                      index   = methods_loadings_,
                      columns = methods_loadings_)
    colors = pd.Series(method_type_).map(method2color).to_numpy()
    ## Clustermap
    ax=sns.clustermap(data=df,cmap='Blues',vmin=0,vmax=1,
                      row_colors=colors, col_colors=colors,square=True,
                      row_cluster=False,col_cluster=False)
    plt.savefig(os.path.join(logdir_loadings,
                             'loadings_corr-{}.svg'.format(ii+1)))
    plt.close()
    ## Dendrogram only
    Z = hierarchy.linkage(df, 'ward')
    _ = hierarchy.dendrogram(Z, leaf_rotation=90, leaf_font_size=8, labels=df.index,
                             color_threshold=0, above_threshold_color='k')
    plt.tight_layout()
    plt.savefig(os.path.join(logdir_loadings,
                             'loadings_corr_dendrogram-{}.svg'.format(ii+1)))
    plt.tight_layout()
    plt.close()
    ## Clustermap with dendrogram
    ax=sns.clustermap(data=df,cmap='Blues',vmin=0,vmax=1,
                      row_linkage=Z, col_linkage=Z,square=True)
    plt.tight_layout()
    plt.savefig(os.path.join(logdir_loadings,
                             'loadings_corr2-{}.svg'.format(ii+1)))
    plt.close()


#### ===========================================================================
#### 2. Gene Emebeddings
#### ===========================================================================

logdir_GeneEmbedding = os.path.join(logdir_analysis,'GeneEmbedding')
os.makedirs(logdir_GeneEmbedding,exist_ok=True)

values_dict = analysis.extract_value(siVAE_result)

#### need change
# gene_embeddings = siVAE_result.get_feature_embeddings()
gene_embeddings = siVAE_result.get_gene_embeddings()
gene_names = siVAE_result.get_model().get_value('var_names')

df_single = df_single.reindex(gene_names)
df_single['Gene'] = df_single.index
df_single['PVE'] = df_single['PVE'].fillna(0)
df_single['Recon Loss'] = df_single['Recon Loss'].fillna(df_single['Recon Loss'].max())

#### ---------------------------------------------------------------------------
#### a. Plot scatterplots
#### ---------------------------------------------------------------------------

X_plot, dim_labels = reduce_dimensions(gene_embeddings, reduced_dimension = 2, method = 'tSNE')

## Set dataframe for plot
df_plot = pd.concat([pd.DataFrame(X_plot), pd.DataFrame(gene_names)], axis = 1)
df_plot.columns = dim_labels + ["Label"]
df_plot.index = gene_names
df_plot['Recon Loss per gene'] = recon_loss_per_gene
df_plot['Dropout'] = dropouts
df_plot['Accuracy'] = 1 - df_plot['Recon Loss per gene']

## Calculate "gene predictive power" by average correlations
corr_mat= np.corrcoef(datah_sample.X.X.transpose())
PVE_corr = pd.DataFrame(np.abs(corr_mat).mean(0),columns=['PVE corr'],
             index=datah_sample.X.var_names)
PVE_corr = PVE_corr.loc[gene_names]
df_plot['PVE_corr'] = PVE_corr
df_plot = pd.concat([df_single,df_plot],axis=1)

df_plot_ = df_plot[df_plot.PVE != 0]
for hue in ['Recon Loss', 'PVE', 'Accuracy', 'Dropout']:
    ax = sns.scatterplot(x = dim_labels[0], y = dim_labels[1], hue = hue, data = df_plot_)
    remove_spines(ax)
    plt.savefig(os.path.join(logdir_GeneEmbedding,'gene_embeedings_scatterplot-{}.svg'.format(hue)))
    plt.close()

ax = sns.scatterplot(x = 'Accuracy', y = 'PVE', hue = 'Dropout', data = df_plot_)
remove_spines(ax)
plt.tight_layout()
plt.savefig(os.path.join(logdir_GeneEmbedding,'PVE vs Accuracy.svg'))
plt.close()

ax = sns.scatterplot(x = 'Dropout', y = 'PVE', data = df_plot_)
remove_spines(ax)
plt.tight_layout()
plt.savefig(os.path.join(logdir_GeneEmbedding,'PVE vs Dropout.svg'))
plt.close()

ax = sns.scatterplot(x = 'Accuracy', y = 'PVE_corr', hue = 'Dropout', data = df_plot_)
remove_spines(ax)
plt.tight_layout()
plt.savefig(os.path.join(logdir_GeneEmbedding,'PVE corr vs Accuracy.svg'))
plt.close()

ax = sns.scatterplot(x = 'PVE', y = 'Dropout', data = df_plot_)
remove_spines(ax)
plt.tight_layout()
plt.savefig(os.path.join(logdir_GeneEmbedding,'PVE vs Dropout.svg'))
plt.close()

#### Scatterplot with Geneset Annotations --------------------------------------
## Need gene_embeddings/gene_names/df_single

import gseapy as gp

logdir_gsea = os.path.join(logdir_analysis,'GSEA')
os.makedirs(logdir_gsea,exist_ok=True)

genesets = {"scsig":"data/MSigDB/scsig.all.v1.0.1.symbols.gmt",
            "Hallmark":"data/MSigDB/h.all.v7.1.symbols.gmt",
            "KEGG":"data/MSigDB/c2.cp.kegg.v7.1.symbols.gmt"}

#### Create data frame where each rows are genes and columns are metadata/stat
X_plot, dim_labels = reduce_dimensions(gene_embeddings, reduced_dimension = 2, method = 'tSNE')
df_plot = pd.concat([pd.DataFrame(X_plot), pd.DataFrame(gene_names)], axis = 1)
df_plot.columns = dim_labels + ["Label"]
df_plot.index = gene_names
df_plot['Recon Loss per gene'] = recon_loss_per_gene
df_plot['Distance from Origin'] = np.square(gene_embeddings).sum(1)

df_plot = pd.concat([df_plot,df_single],axis=1)

#### Set custom grouping of genesets

## {geneset_name: list of genes}
gs_dict = gp.parser.gsea_gmt_parser(genesets['scsig'])

## Filter genesets to Aizarani liver cells
gs_use = [k for k in gs_dict.keys() if 'AIZARANI_LIVER' in k.upper()]
gs_dict = {gs:gs_v for gs,gs_v in gs_dict.items() if gs in gs_use}

## Combine genesets with similar names
gs_name_mapping = {k: "_".join(k.split('_')[3:-1]) for k in gs_dict.keys()}
gs_dict_new = {gs:np.array([]) for gs in np.unique([v for v in gs_name_mapping.values()])}
for gs_name,gs_name2 in gs_name_mapping.items():
    gs_dict_new[gs_name2] = np.union1d(gs_dict_new[gs_name2],gs_dict[gs_name])

## Set selected cell type/group
selected_ct = ('Hepatocytes',
               'Kupffer_Cells',
               'NK_NKT_cells',
               'MHC_II_pos_B')
gs_dict_comb_subset = {k: gs_dict_new[k] for k in selected_ct}

## Get mutually exclusive sets
gs_dict_excl = {}
for gs_name, gs_genes in gs_dict_comb_subset.items():
    for gs_name2, gs_genes2 in gs_dict_new.items():
        if gs_name != gs_name2:
            gs_genes = np.setdiff1d(gs_genes,gs_genes2)
    gs_dict_excl[gs_name] = gs_genes

## Plot setting
plt.rcParams['patch.linewidth'] = 0
plt.rcParams['patch.edgecolor'] = 'none'
plt.rcParams["patch.force_edgecolor"] = False
plt.rcParams['scatter.edgecolors'] = "none"

## Plot annotated genese
df_plot_ = df_plot.copy()
df_plot_ = df_plot_.sort_values('Recon Loss per gene',ascending=True)[:500]
fig_types = ['combined','individual','combined_excl','combined_subset']
gs_dicts  = [gs_dict_new,gs_dict,gs_dict_excl,gs_dict_comb_subset]
name2gs   = {type:gs_dict for type,gs_dict in zip(fig_types,gs_dicts)}

for fig_type,gs_dict_in in name2gs.items():
    figname = os.path.join(logdir_gsea,'gene_embeedings_scatterplot-gs-{}.pdf'.format(fig_type))
    with PdfPages(figname) as pp:
        for gs in list(gs_dict_in.keys()) + ['All','Legend']:
            if gs == 'All':
                df_plot_['Geneset'] = None
                for gs_,genes in gs_dict_in.items():
                    df_plot_['Geneset'].loc[np.isin(df_plot_.index,genes)] = gs_
            elif gs == 'Legend':
                pass
            else:
                df_plot_['Geneset'] = 'None'
                gs_   = gs
                genes = gs_dict_in[gs]
                df_plot_['Geneset'].loc[np.isin(df_plot_.index,genes)] = gs_
                df_plot_['Geneset_order'] = df_plot_['Geneset'] != 'None'
                df_plot_ = df_plot_.sort_values(['Geneset_order','Geneset'])
            ax = sns.scatterplot(x = dim_labels[0], y = dim_labels[1], hue = 'Geneset', data = df_plot_,
                                 edgecolor='black',linewidth=0.2,s=150)
            remove_spines(ax)
            if gs == 'Legend':
                # lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                legend = plt.legend(edgecolor='black')
                legend.get_frame().set_alpha(1)
                plt.tight_layout()
            else:
                ax.legend_.remove()
            plt.title(gs)
            pp.savefig()
            plt.clf()
    plt.close()

#### Cell embeddings Scatterplot with Geneset Annotations ----------------------

#### need change
cell_embeddings = siVAE_result.get_cell_embeddings()
cell_labels = siVAE_result.get_model().get_value('labels')
X_plot, dim_labels = reduce_dimensions(cell_embeddings, reduced_dimension = 2, method = 'tSNE')
df_plot = pd.concat([pd.DataFrame(X_plot), pd.DataFrame(cell_labels)], axis = 1)
df_plot.columns = dim_labels + ["Label"]
df_plot.index = cell_labels

## Set cell type groupings
cell_labels_dict = {k:k for k in np.unique(cell_labels)}
cell_labels_dict['Hepatocyte'] = 'Hepatocytes'
cell_labels_dict['Kupffer Cell'] = 'Kupffer_Cells'
cell_labels_dict['NK'] = 'NK_NKT_cells'
cell_labels_dict['Mono-NK'] = 'NK_NKT_cells'
cell_labels_dict['Mac NK'] = 'NK_NKT_cells'
# cell_labels_dict['Fibroblast'] = 'Stellate_cells'
# cell_labels_dict['HSC/MPP'] = 'Stellate_cells'
cell_labels_dict['pro B cell'] = 'MHC_II_pos_B'
cell_labels_dict['pre B cell'] = 'MHC_II_pos_B'
cell_labels_dict['pre pro B cell'] = 'MHC_II_pos_B'
cell_labels_2 = [cell_labels_dict[c] for c in cell_labels]

df_plot['Label'] = cell_labels_2
df_plot_ = df_plot
df_plot_ = df_plot_[np.isin(df_plot_['Label'],selected_ct)]
df_plot_['Label'] = df_plot_['Label'].astype('category')
df_plot_['Label'].cat.set_categories(selected_ct,inplace=True)
df_plot_.sort_values('Label',inplace=True)
ax = sns.scatterplot(x = dim_labels[0], y = dim_labels[1], hue = 'Label', data = df_plot_,
                     edgecolor='black',linewidth=0.2)
remove_spines(ax)
ax.legend_.remove()
plt.savefig(os.path.join(logdir_gsea,'cell_embeddings_subset.svg'))
plt.close()

#### Calculate Correlation between cell types
genes_subset = np.unique(np.concatenate(list(gs_dict_excl.values())))
X = siVAE_result.get_model().get_value('reconstruction')[0]
X_corr = np.corrcoef(X)
input_ct = selected_ct
list1 = []
for ct1 in input_ct:
    list2 = []
    for ct2 in input_ct:
        idx_1 = np.where(np.isin(cell_labels_2,ct1))[0]
        idx_2 = np.where(np.isin(cell_labels_2,ct2))[0]
        m = np.abs(X_corr[idx_1][:,idx_2]).mean()
        list2.append(m)
    list1.append(list2)

overlaps = np.array([np.sum(np.isin(cell_labels_2,ct1)) for ct1 in input_ct])
overlaps = overlaps.reshape(-1,1)
overlaps = np.matmul(overlaps,overlaps.transpose())

df_corr = pd.DataFrame(np.array(list1),columns=input_ct,index=input_ct)
corr = np.array(list1)
## Take into account number of each cell type
mean_corr = (overlaps * corr)[np.triu_indices(len(input_ct),1)][:-1].sum() \
            / overlaps[np.triu_indices(len(input_ct),1)][:-1].sum()

#### Perform GSEA --------------------------------------------------------------
# methods_loadings = ['siVAE']
# all_loadings = [gene_embeddings.transpose()]
# for loading_method, loadings in zip(methods_loadings,all_loadings):
#     print(loadings.shape)
#     for dim,loading in enumerate(loadings):
#         df = pd.DataFrame(loading)
#         df.index = gene_names
#         for gs_name,gs_dir in genesets.items():
#             outdir = os.path.join(logdir_gsea,
#                                   'method-{}'.format(loading_method),
#                                   'dim-{}'.format(dim),
#                                   "GSEA-{}".format(gs_name))
#             pre_res = gp.prerank(rnk=df, gene_sets=gs_dir, outdir=outdir)
#             df_pre = pre_res.res2d.sort_values('fdr')
#             df_plot['Geneset'] = 'None'
#             for gs_,genes in zip(df_pre.index[:5][::-1],df_pre.genes[:5][::-1]):
#                 genes_list = genes.split(';')
#                 df_plot['Geneset'].loc[np.isin(df_plot.index,genes_list)] = gs_
#             ##
#             ax = sns.scatterplot(x = dim_labels[0], y = dim_labels[1], hue = 'Geneset', data = df_plot)
#             plt.savefig(os.path.join(outdir,'gene_embeedings_scatterplot-{}.pdf'.format(gs_name)))
#             plt.close()


## -----------------------------------------------------------------------------
## b/c. Load similarity matrix for identfication of central/neighborhood genes
## -----------------------------------------------------------------------------

#### Pull up gene score, relative score for how important gene is to predicting rest of genome

#### need move
def extract_similarity_matrix(siVAE_result, gene_names, do_siVAE=True, do_FA=True, do_layers=True,
                              method_DE=None, method='',category='', grn_mat_dict={}, method2category={}):
    """
    do_FA: extract loadings inferred from feature attribution
    """
    ##
    if category == '':
        category=method
    method_name = method
    ## Extract siVAE Loadings
    if do_siVAE:
        #### need change
        # v_mu = siVAE_result.get_feature_embeddings().transpose()
        v_mu = siVAE_result.get_gene_embeddings().transpose()
        df_mat = -loadings2dist(v_mu.transpose(),gene_names)
        grn_mat_dict[method_name] = df_mat
        method2category[method_name] = category
    ## Extract Loadings from Feature Attribution
    if do_FA:
        infer_FA_method=1
        scores, method_DE = get_feature_attributions(siVAE_result)
        scores = scores['decoder']
        FA_loadings = analysis.infer_FA_loadings(np.swapaxes(scores,1,2),
                                                       method=infer_FA_method)
        ## Feature attributions loadings
        FA_loadings_ = np.swapaxes(FA_loadings,1,2)
        FA_loadings = FA_loadings[::-1]
        method_DE = method_DE[::-1]
        method_FA = ["{}-{}".format(method,m) for m in method_DE]
        for FA_method,loadings in zip(method_FA,FA_loadings_):
            df_mat = -loadings2dist(loadings,gene_names,gene_names)
            grn_mat_dict[FA_method] = df_mat
            method2category[FA_method] = category
    if do_layers:
        for ii, loadings in enumerate(siVAE_result.get_model().get_value('decoder_layers')['feature']):
            layer_name="{}-L{}(n={})".format(method,ii+2,loadings.shape[-1])
            df_mat = -loadings2dist(loadings,gene_names,gene_names)
            grn_mat_dict[layer_name] = df_mat
            method2category[layer_name] = category
    return grn_mat_dict, method2category


def loadings2dist(loadings,gene_names,gene_names2=None):
    """
    Input: gene embeddings/loadings
    Return: n_gene X n_gene matrix showing distance between genes
    """
    df_mat = pd.DataFrame([np.linalg.norm(loadings-g,axis=1) for g in loadings],
                          index = gene_names,
                          columns = gene_names)
    if gene_names2 is not None:
        df_mat = df_mat.reindex(gene_names2).transpose().reindex(gene_names2).transpose().fillna(0)
    return df_mat


from siVAE.model.output.degree_centrality import calculate_degree_centrality

## Calculate degree centrality for siVAE
result = siVAE_result.get_model()
y      = result.get_value('reconstruction')[0]
y_pred = result.get_value('reconstruction')[1]
recon_loss_per_gene = np.square(y - y_pred).mean(0)
degree_centrality = calculate_degree_centrality(recon_loss=recon_loss_per_gene)
df_DC = pd.DataFrame({'siVAE': degree_centrality})
df_DC.index = gene_names
df_DC['PVE'] = df_single.loc[gene_names].PVE

## Calculate GRN similarity matrix for siVAE
# grn_mat_dict = {}
## Method 1: distance away in gene embedding space
# df_mat = -loadings2dist(gene_embeddings,gene_names)
## Method 2: Feature attribution
# df_mat = pd.DataFrame(FA,index=gene_names,columns=gene_names)

grn_mat_dict, method2category = extract_similarity_matrix(siVAE_result,
                                                         gene_names=gene_names,
                                                         do_siVAE=True,
                                                         do_FA=True,
                                                         do_layers=True,
                                                         method='siVAE',
                                                         category='siVAE',
                                                         grn_mat_dict={},
                                                         method2category={})

for model, logdir_ in siVAE_models_dict.items():
    siVAE_result_dir = os.path.join(logdir_,"siVAE_result.pickle")
    siVAE_result_ = util.load_pickle(siVAE_result_dir)
    if 'scVI' == model:
        grn_mat_dict,method2category = extract_similarity_matrix(siVAE_result_,
                                                                 gene_names=gene_names,
                                                                 do_siVAE=False,
                                                                 do_FA=True,
                                                                 do_layers=False,
                                                                 method=model,
                                                                 category=model,
                                                                 grn_mat_dict=grn_mat_dict,
                                                                 method2category=method2category
                                                                 )
    else:
        grn_mat_dict,method2category = extract_similarity_matrix(siVAE_result_,
                                                                 gene_names=gene_names,
                                                                 do_siVAE=True,
                                                                 do_FA=True,
                                                                 do_layers=True,
                                                                 method=model,
                                                                 category=model,
                                                                 grn_mat_dict=grn_mat_dict,
                                                                 method2category=method2category
                                                                 )

## LDVAE loadings
logdir_scVI = logdir_dict['LDVAE']
if logdir_scVI is not None:
    df_ = pd.read_csv(os.path.join(logdir_scVI,'X_loadings.csv'),index_col=0)
    loadings = df_.values
    gene_names2 = df_.index
    df_mat = -loadings2dist(loadings,gene_names,gene_names2)
    grn_mat_dict['LDVAE'] = df_mat
    method2category['LDVAE'] = "scVI"


## Load GRN adjacency matices for internal layers
def load_GRN(logdir_GRN, gene_names=None):
    df_GRN = pd.read_csv(logdir_GRN,index_col=0).abs()
    df_GRN = df_GRN.reindex(gene_names).transpose().reindex(gene_names).transpose().fillna(0)
    df_dc = df_GRN.mean(1)
    return df_GRN, df_dc

logdir_GRN = logdir_dict['GRN']

for input_type in os.listdir(logdir_GRN):
    logdir_input = os.path.join(logdir_GRN,input_type)
    category_name = input_type + ' (GRN)'
    for method in os.listdir(logdir_input):
        logdir_method = os.path.join(logdir_input,method,'adjmat.csv')
        df_GRN, df_dc = load_GRN(logdir_method)
        method_name="{}-{}".format(input_type,method.upper())
        ## Add to GRN dicts
        grn_mat_dict[method_name]    = df_GRN
        df_DC[method_name]           = df_dc
        method2category[method_name] = category_name


# ## Load siVAE with varying number of layers
# category_name='siVAE (LE)'
# LE = 'LE_dim-' + logdir_siVAE.split('LE_dim-')[1].split("/")[0]
# for LE_dim in [2,16,64]:
#     ##
#     LE_ = 'LE_dim-'+str(LE_dim)
#     logdir_ = logdir_siVAE.replace(LE,LE_)
#     siVAE_result_dir = os.path.join(logdir_,"siVAE_result.pickle")
#     # Load
#     siVAE_result_ = util.load_pickle(siVAE_result_dir)
#     gene_embeddings_ = siVAE_result_['model']['latent_embedding']['feature']
#     df_mat = -loadings2dist(gene_embeddings_ ,gene_names)
#     ##
#     method_name_ = 'siVAE-(LE={})'.format(LE_dim)
#     grn_mat_dict[method_name_] = df_mat
#     method2category[method_name_] = category_name
#     ##
#     del siVAE_result_
#     gc.collect()

methods = list(dict.fromkeys(method2category.values()))
colors_dict = {key:color for key,color in zip(methods,sns.color_palette())}

#### ---------------------------------------------------------------------------
#### b. Identification of central genes
#### ---------------------------------------------------------------------------

#### Create gene predictive power vs degree centrality -------------------------
logdir_DC = os.path.join(logdir_analysis,'DegreeCentrality')
os.makedirs(logdir_DC,exist_ok=True)

## Correlation
df_DC_rank = pd.DataFrame(df_DC)
df_DC_rank[:] = (df_DC_rank.to_numpy()).argsort(axis=0).argsort(axis=0)
df_DC_rank['PVE'] = df_single['PVE'].to_numpy()

df_plot = df_DC[df_DC['PVE'] != 0]
df_plot = df_plot.melt('PVE', var_name = 'Methods', value_name = 'Gene Degree Centrality')

reorder = ['siVAE','MRNET','GRNBOOST2','CLR','ARACNE']
cat = pd.Categorical(df_plot['Methods'], categories=reorder, ordered=True)
df_plot['Methods'] = cat
df_plot = df_plot.sort_values(by='Methods')

sns.set_style("ticks")
g = sns.FacetGrid(df_plot, col = 'Methods', sharex=False, sharey=True)
g.map(sns.scatterplot, 'Gene Degree Centrality', 'PVE')
g.set(ylim=(0,None),xlim=(0,None))
for ii,ax in enumerate(g.axes[0]):
    remove_spines(ax,num_ticks=3,show_legend=False)
    if ii != 0:
        ax.tick_params(labelleft=False,left=False)

plt.tight_layout()
plt.savefig(os.path.join(logdir_DC,'PVE_vs_GeneImportance.svg'))
plt.close()

## Calculate correlation between Predicted Degree Centrality vs Predictive Power (PVE)
df_plot = df_DC[df_DC['PVE'] != 0]
methods = [ii for ii in df_plot.columns if ii != 'PVE']
df_spearmanr = pd.DataFrame([stats.spearmanr(df_plot[m],df_plot['PVE']) for m in methods],
                            index = methods)
df_spearmanr.to_csv(os.path.join(logdir_DC,'Corr_GTvsPredicted_DC.csv'))

## Plot only genes with top 100 PVE values -------------------------------------
n_top = 100
df_plot = df_DC[df_DC['PVE'] != 0]
df_plot = df_plot.sort_values('PVE',ascending=False).iloc[:n_top]
df_plot = df_plot.melt('PVE', var_name = 'Methods', value_name = 'Gene Degree Centrality')
g = sns.FacetGrid(df_plot,col = 'Methods', sharex=False)
g.map(sns.scatterplot, 'Gene Degree Centrality', 'PVE')
for ax in g.axes[0]:
    remove_spines(ax,num_ticks=4)

plt.tight_layout()
plt.savefig(os.path.join(logdir_DC,'PVE_vs_GeneImportance_100.svg'))
plt.close()

#### PVE vs Gene Importance Top AUC --------------------------------------------
df_plot = df_DC[df_DC['PVE'] != 0]
for n_top in [50,'all']:
    df_list = []
    for col in df_plot.drop('PVE',axis=1).columns:
        if n_top == 'all':
            n_top_num = len(df_plot)
        else:
            n_top_num = n_top
        df_ = df_plot.sort_values(col,ascending=False).iloc[:n_top_num]
        df_ = pd.DataFrame(df_['PVE'])
        df_['Rank'] = (np.arange(n_top_num)+1)
        df_['Method'] = col
        df_['Cumulative sum of PVE per gene'] = df_['PVE'].cumsum()
        df_list.append(df_)
    #
    df_plot2 = pd.concat(df_list,axis=0)
    df_plot2 = df_plot2.reset_index()
    ax = sns.lineplot(data = df_plot2, x='Rank', y='Cumulative sum of PVE per gene',hue='Method',linewidth=3)
    remove_spines(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(logdir_DC,'PVE_Cum vs Rank_{}.svg').format(n_top))
    plt.close()

n_top = 50
df_ = df_plot2[df_plot2.Rank <= n_top].groupby('Method').mean()
df_ = df_.reindex(['siVAE','ARACNE','MRNET','CLR','GRNBOOST2'])
ax = sns.barplot(data=df_,x=df_.index,y='PVE')
remove_spines(ax,show_legend=False,xaxis=False)
plt.tight_layout()
plt.savefig(os.path.join(logdir_DC,'PVE_Sum_{}.svg').format(n_top))
plt.close()

#### correlation matrix --------------------------------------------------------
def max_spearmanr(matrix,axis=1,neg=False):
    if axis == 0:
        matrix = matrix.transpose()
    if neg:
        matrix2 = -matrix
    else:
        matrix2 = matrix
    results = [[stats.spearmanr(m1,m2) for m2 in matrix2] for m1 in matrix]
    corr   = np.array([[r[0] for r in rr] for rr in results])
    pvalue = np.array([[r[1] for r in rr] for rr in results])
    return corr,pvalue

corr_list = np.array([stats.spearmanr(df_DC[col],df_DC['PVE']) for col in df_DC])
df_corr = pd.DataFrame(corr_list,
                       columns=['Spearman Correlation','P-value'],
                       index=df_DC.columns)
df_corr['Method'] = df_corr.index
df_corr.to_csv(os.path.join(logdir_DC,'hub_corr.csv'),index=False,header=True)


#### ---------------------------------------------------------------------------
#### c. Identification of neigborhood genes
#### ---------------------------------------------------------------------------

logdir_neighborhood = os.path.join(logdir_analysis,'NeighborhoodPred')
os.makedirs(logdir_neighborhood,exist_ok=True)

num_neighbors = 50

#### Filter genes based on minimum number of neighbors across method -------
def cor_to_adj(X, top_quantile = 0.1, diag = False):
    if X.shape[0] != X.shape[1]:
        raise Exception('X must be a square matrix')
    cutoff_index = int(X.size * top_quantile)
    cutoff = np.array(X).flatten()
    cutoff.sort()
    cutoff_value = cutoff[-cutoff_index]
    adj = (X >= cutoff_value).astype('int')
    return adj

from functools import reduce
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

## Check scatterplot of gene embeddings to see if genes_neighborhood clusters
df_adj = df_DC.iloc[:,:-1]
genes_neighborhood = reduce(np.intersect1d,[v.sort_values(ascending=False)[:500].index for k,v in df_adj.iteritems()])
X_red, _ = reduce_dimensions(gene_embeddings, reduced_dimension = 2, method = 'PCA')
df_plot_ = pd.DataFrame(X_red,columns=['siVAE-1','siVAE-2'])
df_plot_['Highly Connected Genes'] = np.isin(df_adj.index,genes_neighborhood)
ax = sns.scatterplot(x = 'siVAE-1', y = 'siVAE-2', hue = 'Highly Connected Genes', data = df_plot_,
                     edgecolor='black',linewidth=0.2)
plt.savefig(os.path.join(logdir_neighborhood,'gene_embeddings-neighborhood.pdf'))
plt.close()

#
# exp_train,exp_test = train_test_split(datah_sample.X.X,test_size=0.1,
#                                       stratify=datah_sample.X.obs['Cell Type'])
#
# corr_mat = np.corrcoef(datah_sample.X.X.transpose())
# df_corr = pd.DataFrame(np.abs(corr_mat),
#                        index=datah_sample.X.var_names,
#                        columns=datah_sample.X.var_names)
#
# corr_method = []
# for method,df_mat in grn_mat_dict.items():
#     df_mat = df_mat.loc[genes_neighborhood]
#     corr_v = []
#     for g,v in df_mat.iterrows():
#         v = v[v.index != g]
#         genes_v = v.sort_values(ascending=False)[:3].index.to_numpy()
#         corr_v.append(df_corr.loc[genes_v].mean().mean())
#     corr_method.append(corr_v)
#
# np.array(corr_method).mean(1)

#### Overlaps between neighborhood genes across methods ------------------------
## Need grn_mat_dict

from functools import reduce
genes_list = []
for method,df_mat in grn_mat_dict.items():
    genes_ = []
    for g,v in df_mat.iteritems():
        v = v[v.index != g]
        genes_v = v.sort_values(ascending=False)[:num_neighbors].index.to_numpy()
        genes_.append(genes_v)
    genes_list.append(genes_)

## Get overlaps
neighbors_per_method = np.array(genes_list)

overlaps = [[[len(np.intersect1d(z1,z2))/len(np.union1d(z1,z2)) for z1,z2 in zip(g1,g2)] for g2 in neighbors_per_method] for g1 in neighbors_per_method]
overlaps = np.array(overlaps) # n_method x n_method x n_genes
# overlaps = overlaps / (neighbors_per_method.shape[-1] * 2 - overlaps)
df_overlap = pd.DataFrame(overlaps.mean(-1),
                          columns=grn_mat_dict.keys(),
                          index=grn_mat_dict.keys())
df_overlap.to_csv(os.path.join(logdir_neighborhood,'jaccard_index.csv'))

#### Plot heatmaps for all methods

## Set custom color mapping
cmap=ListedColormap(np.linspace([256,256,256],[155,41,97])/256)

sns.set(font_scale=0.5)
colors = [colors_dict[method2category[m]] for m in df_overlap.index]
ax=sns.clustermap(data=df_overlap,cmap=cmap,vmin=0,vmax=1,
                  row_cluster=False,col_cluster=False,
                  row_colors=colors, col_colors=colors,square=True)
# ax=sns.clustermap(data=df_overlap,cmap='Blues',vmin=0,vmax=1,
#                   row_colors=colors, col_colors=colors,square=True)
plt.savefig(os.path.join(logdir_neighborhood,'overlaps_in_neighborhood.svg'))
plt.close()

#### Subset the methods
# subset_plot = ['siVAE','siVAE-0','siVAE-l2 (GRN)','scVI-DeepLIFT','LDVAE',
#                'MRNET','CLR','GRNBOOST2','ARACNE']
subset_plot = ['siVAE','siVAE-0','siVAE-l2-GRNBOOST2','LDVAE',
               'gene_relevance_input-MRNET',
               'gene_relevance_input-CLR',
               'gene_relevance_input-GRNBOOST2',
               'gene_relevance_input-ARACNE']

from matplotlib.colors import ListedColormap
df_overlap2 = df_overlap.reindex(subset_plot).transpose().reindex(subset_plot)
colors = [colors_dict[method2category[m]] for m in df_overlap2.index]
colors = [sns.color_palette()[0]] * 4 + [sns.color_palette()[1]] * 4

## Set custom color gradient palette for heatmap
cbar_pos=(0.02, 0.8, 0.05, 0.18)
ax=sns.clustermap(data=df_overlap2,cmap=cmap,vmin=0,vmax=1,
                  row_cluster=False,col_cluster=False,
                  row_colors=colors, col_colors=colors,square=True, cbar_pos=cbar_pos)
# plt.tight_layout()
plt.savefig(os.path.join(logdir_neighborhood,'overlaps_in_neighborhood2.svg'))
plt.close()

## Create dendrogram
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
Z = hierarchy.linkage(df_overlap2, 'ward', optimal_ordering=True)
fig,ax = plt.subplots()
ax.set_facecolor("white")
hierarchy.dendrogram(Z, ax=ax, leaf_rotation=90, leaf_font_size=8,
                     count_sort=True,
                     distance_sort=True,
                     labels=df_overlap2.index)
plt.tight_layout()
plt.savefig(os.path.join(logdir_neighborhood,'overlaps_in_neighborhood2_dendrogram.svg'))
plt.close()

ax=sns.clustermap(data=df_overlap2,cmap=cmap,vmin=0,vmax=1,
                  row_linkage=Z, col_linkage=Z,square=True,
                  row_colors=colors, col_colors=colors)
plt.tight_layout()
plt.savefig(os.path.join(logdir_neighborhood,'overlaps_in_neighborhood2_w_dendrogram.svg'))
plt.close()


#### Pairwise-corr between neighborhood genes across methods -------------------

logdir_corr = os.path.join(logdir_neighborhood,"pairwise_corr")
os.makedirs(logdir_corr,exist_ok=True)

corr_mat= np.corrcoef(datah_sample.X.X.transpose())
gnames = datah_sample.X.var_names
corr_df = pd.DataFrame(corr_mat,index=gnames,columns=gnames)
df_index = pd.DataFrame(gnames)
df_index['idx'] = df_index.index
df_index.index = gnames

## Subset
subdict = grn_mat_dict
methods = subset_plot
subdict = {m:grn_mat_dict[m] for m in methods}

import time
times = []
pairwise_1 = []
for method,df_mat1 in grn_mat_dict.items():
    pairwise_2 = []
    for method2,df_mat2 in grn_mat_dict.items():
        pairwise = 0
        start_time = time.time()
        corr_file = os.path.join(logdir_corr,"{}-{}.npy".format(method,method2))
        if not os.path.isfile(corr_file) or redo:
            for (g1,v1),(g2,v2) in zip(df_mat1.iteritems(), df_mat2.iteritems()):
                ##
                v1 = v1[v1.index != g1]
                genes_v = v1.nlargest(num_neighbors).index.to_numpy()
                genes_idx1 = df_index.loc[genes_v]['idx'].to_numpy()
                ##
                v2 = v2[v2.index != g1]
                genes_v = v2.nlargest(num_neighbors).index.to_numpy()
                genes_idx2 = df_index.loc[genes_v]['idx'].to_numpy()
                ##
                corr_mat_ = corr_mat[genes_idx1][:,genes_idx2]
                pairwise += np.abs(corr_mat_)
            pairwise = pairwise / len(df_mat1) # Divide for average
            np.save(corr_file,pairwise,allow_pickle=True)
        else:
            pairwise = np.load(corr_file)
        pairwise_2.append(pairwise)
        times.append(time.time() - start_time)
    pairwise_1.append(pairwise_2)

#### Plots
cmap = ListedColormap(np.linspace([256,256,256],[35,133,50])/256)

# ## Plots
# n_method = len(subdict)
# colors = np.repeat(np.array(sns.color_palette("tab10")[:n_method]),num_neighbors,axis=0)
# pairwise_matrix = np.array(pairwise_1)
# n_col=n_row=num_neighbors * n_method
# pairwise_matrix = pairwise_matrix.swapaxes(1,2).reshape(n_row,n_col)
# g = sns.clustermap(data = pairwise_matrix, cmap='Blues', vmax=1, vmin=0,
#                row_cluster=False, col_cluster=False,
#                row_colors = colors, col_colors = colors)
# _ = g.ax_heatmap.tick_params(left=False, bottom=False,right=False,top=False)
# _ = g.ax_heatmap.set(xticklabels=[],yticklabels=[])
# plt.savefig(os.path.join(logdir_neighborhood,'pairwise_heatmap_genes.pdf'))
# plt.close()

## Mean
methods = [ k for k in grn_mat_dict.keys()]
pairwise_matrix = np.array(pairwise_1).mean(-1).mean(-1)
df_plot = pd.DataFrame(pairwise_matrix,
                       index=methods,
                       columns=methods)
df_plot.to_csv(os.path.join(logdir_neighborhood,'MeanCorrelation.csv'))
colors = [colors_dict[method2category[m]] for m in df_plot.index]
ax=sns.clustermap(data=df_plot,cmap=cmap,
                  row_cluster=False,col_cluster=False,
                  row_colors=colors, col_colors=colors,square=True)
plt.savefig(os.path.join(logdir_neighborhood,'pairwise_heatmap_genes_mean.svg'))
plt.close()

methods = [ k for k in subdict.keys()]
df_overlap2 = df_plot.reindex(subset_plot).transpose().reindex(subset_plot)

## Create custom palette
# colors = [colors_dict[method2category[m]] for m in df_overlap2.index]
colors = [sns.color_palette()[0]] * 5 + [sns.color_palette()[1]] * 4
ax=sns.clustermap(data=df_overlap2,cmap=cmap,
                  row_cluster=False,col_cluster=False,
                  row_colors=colors, col_colors=colors,square=True)
plt.savefig(os.path.join(logdir_neighborhood,'pairwise_heatmap_genes_mean2.svg'))
plt.close()

from scipy.cluster import hierarchy

Z = hierarchy.linkage(df_overlap2, 'ward')
hierarchy.dendrogram(Z, leaf_rotation=90, leaf_font_size=8, labels=df_overlap2.index)
plt.savefig(os.path.join(logdir_neighborhood,'pairwise_heatmap_genes_mean2_dendrogram.svg'))
plt.close()

#### need change
## Average
idx_not_LDVAE = np.where(np.array(list(grn_mat_dict.keys())) != 'LDVAE')[0]
idx_not_ARACNE = np.where(np.array(list(grn_mat_dict.keys())) != 'ARACNE')[0]
idx_not = np.intersect1d(idx_not_LDVAE,idx_not_ARACNE)
idx_dict = {'All_no_LDVAE': idx_not}
idx_dict['GRN'] = np.where(np.array(np.isin(list(grn_mat_dict.keys()),['GRNBOOST2','MRNET','CLR'] )))[0]
idx_dict['DR']  = np.arange(0,6)

comparisons = []
comparisons.append(['GRN','DR'])
comparisons.append(['DR','DR'])
comparisons.append(['GRN','GRN'])

stats_comparisons = []
for t1,t2 in comparisons:
    stats_list = []
    for mat_ in [df_plot.to_numpy()]:
        mat_ = mat_[np.array(idx_dict[t1])][:,np.array(idx_dict[t2])]
        if t1==t2:
            mat_ = np.triu(mat_).reshape(-1)
            mat_ = mat_[mat_ != 0]
        else:
            mat_ = mat_.reshape(-1)
        stats_list.append(mat_.mean())
        stats_list.append(np.median(mat_))
    stats_comparisons.append(stats_list)

comparisons_str = ["{} vs {}".format(c1,c2) for c1,c2 in comparisons]
df_stats = pd.DataFrame(stats_comparisons,
                        columns = ['Coef Mean','Coef Median'],
                        index = comparisons_str)


#### Predictive Accuracy of neighborhood genes for target gene -----------------

#### tf
from siVAE.data import data_handler as dh
from siVAE.model import VAE
import tensorflow as tf
logging.getLogger().setLevel(logging.INFO)

gpu_device = '0'
os.environ["CUDA_VISIBLE_DEVICES"]  = gpu_device
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
config.allow_soft_placement = True
config.intra_op_parallelism_threads = 5
config.inter_op_parallelism_threads = 5

graph_args = {'LE_dim'       : 5,
              'architecture' : '5-5-5-0-2',
              'config'       : config,
              'iter'         : 3000,
              'mb_size'      : 0.2,
              'l2_scale'     : 1e-2,
              'dataAPI'      : False,
              'tensorboard'  : False,
              'batch_norm'   : False,
              'keep_prob'    : 0.9,
              'log_frequency': 100,
              'learning_rate': 1e-3,
              "early_stopping"   : False,
              "validation_split" : 0,
              "decay_rate"       : 0.9,
              "decay_steps"      : 1000,
              'activation_fun'   : tf.nn.relu,
              'l2_scale_final'     : 0}

datadir = os.path.join(logdir_exp,'data_dict_tf.pickle')
data_dict = util.load_pickle(datadir)
datah_sample  = data_dict['sample']

architecture = '20-20-20-5-0-3'
LE_dim = 5

datah = dh.data_handler(X=datah_sample.X[:,:num_neighbors],y=datah_sample.X[:,[1]])
datah.split_index_list = datah_sample.split_index_list
h_dims = architecture.replace('LE',str(LE_dim))
h_dims = [int(dim) for dim in h_dims.split("-")]
datah.index_latent_embedding = int(h_dims.pop())
datah.h_dims = h_dims

#### Test
datah_ = datah_sample.X[:,np.isin(datah_sample.X.var_names,grn_mat_dict['siVAE'].index)]

## Additional tests
# grn_mat_dict['Linear']    = pd.DataFrame(np.abs(np.corrcoef(datah_.X.transpose())),
#                                          index=datah_.var_names,columns=datah_.var_names)
# grn_mat_dict['NegLinear'] = -grn_mat_dict['Linear']
# method2category['Linear'] = 'Test'
# method2category['NegLinear'] = 'Test'
# grn_mat_dict['siVAE2'] = grn_mat_dict['siVAE']
# grn_mat_dict['siVAE3'] = grn_mat_dict['siVAE']
# grn_mat_dict['siVAE4'] = grn_mat_dict['siVAE']
# grn_mat_dict['siVAE5'] = grn_mat_dict['siVAE']
# grn_mat_dict['siVAE6'] = grn_mat_dict['siVAE']
# grn_mat_dict['siVAE7'] = grn_mat_dict['siVAE']
# method2category['siVAE2'] = 'Test2'
# method2category['siVAE3'] = 'Test2'
# method2category['siVAE4'] = 'Test2'
# method2category['siVAE5'] = 'Test2'
# method2category['siVAE6'] = 'Test2'
# method2category['siVAE7'] = 'Test2'
# colors_dict['Test2'] = colors_dict['siVAE']

scores_list = []
logdir_n_neighbors = os.path.join(logdir_neighborhood,'test2','Num_Neighbors-{}'.format(num_neighbors))

for method,df_mat in grn_mat_dict.items():
    dir_df = os.path.join(logdir_n_neighbors,'scores_{}.csv'.format(method))
    df_mat = df_mat.loc[:,genes_neighborhood]
    if os.path.exists(dir_df):
        df_score=pd.read_csv(dir_df,index_col=0)
        df_score.Method = method
        df_score.to_csv(dir_df)
    else:
        scores_train = []
        scores_test  = []
        for g,v in df_mat.iteritems():
            v = v[v.index != g]
            genes_v = v.sort_values(ascending=False)[:num_neighbors].index.to_numpy()
            gnames = datah_sample.X.var_names
            ##
            datah.X = datah_sample.X[:,np.isin(gnames,genes_v)]
            datah.y = datah_sample.X[:,np.isin(gnames,g)]
            datah.create_dataset(kfold_idx=0)
            ##
            tf.compat.v1.reset_default_graph()
            graph_args['logdir_tf'] = os.path.join(logdir_n_neighbors,
                                                   'Method-{}'.format(method),
                                                   'Gene-{}'.format(g))
            os.makedirs(graph_args['logdir_tf'],exist_ok=True)
            tf.reset_default_graph()
            sess = tf.Session()
            print('model')
            model = VAE.AutoEncoder(data_handler = datah, random_seed = 0,
                                    isVAE = False, name = 'VAE',
                                    decoder_var='deterministic',**graph_args)
            print('build')
            model.build_model(reset_graph = False)
            print('train')
            result = model.train(sess,initialize=True)
            sess.close()
            ##
            score_train = result.get_value('losses')['train'][2]
            score_test  = result.get_value('losses')['test'][2]
            scores_train.append(score_train)
            scores_test.append(score_test)
            gc.collect()
        df_score = pd.DataFrame({'Gene': df_mat.columns,
                                 'Train': scores_train,
                                 'Test': scores_test},
                                 index = df_mat.columns)
        print(method)
        df_score['Method'] = method
        df_score.to_csv(dir_df)
    df_score['Category'] = method2category[method]
    scores_list.append(df_score)

# df_scores = pd.DataFrame([sc.mean() for sc in scores_list],index=grn_mat_dict.keys())
# df_scores = pd.DataFrame([sc[np.isin(sc.index,genes_neighborhood)].mean() for sc in scores_list],index=grn_mat_dict.keys())
# genes_neighborhood2 = reduce(np.intersect1d,[v.sort_values(ascending=False)[:500].index for k,v in df_adj.iteritems()])
# df_scores = pd.DataFrame([sc[np.isin(sc.index,genes_neighborhood2)].mean() for sc in scores_list],index=grn_mat_dict.keys())

logdir_ = os.path.join(logdir_n_neighbors)
os.makedirs(logdir_,exist_ok=True)

df_plot = pd.concat(scores_list)
df_plot.to_csv(os.path.join(logdir_,'Pred_NumNeighbors.csv'))
df_plot = pd.read_csv(os.path.join(logdir_,'Pred_NumNeighbors.csv'),index_col=0)

df_plot['Recon Loss'] = df_plot['Test']
df_plot['Accuracy'] = (1 - df_plot['Test'])

#### Statistics ----------------------------------------------------------------
from sklearn.preprocessing import scale

df_plot_ = df_plot.iloc[np.isin(df_plot.index,df_plot[df_plot.Method=='siVAE'].index)]
df_plot_ = df_plot_[df_plot.Category != 'Test']
acc = df_plot_.Accuracy.to_numpy().reshape(-1,len(np.unique(df_plot_.index)))
acc_scaled = scale(acc)
# acc_scaled = np.argsort(acc,0)
# coefs = np.corrcoef(acc_scaled)
coefs = np.array([[stats.pearsonr(m1,m2)[0] for m1 in acc_scaled] for m2 in acc_scaled])
pvals = np.array([[stats.pearsonr(m1,m2)[1] for m1 in acc_scaled] for m2 in acc_scaled])

df_methods = {m: df_plot_[df_plot_.Method == m].reindex(scores_list[11].index) for m in pd.unique(df_plot_['Method'])}

# coefs = np.array([[stats.spearmanr(df1.Accuracy_scaled,df2.Accuracy_scaled)[0] for _,df2 in df_methods.items()] for _,df1 in df_methods.items()])
# pvals = np.array([[stats.spearmanr(df1.Accuracy_scaled,df2.Accuracy_scaled)[1] for _,df2 in df_methods.items()] for _,df1 in df_methods.items()])

df_ = pd.DataFrame(coefs,columns=df_methods.keys(),index=df_methods.keys())

sns.set(font_scale=0.5)
colors = [colors_dict[method2category[m]] for m in df_.index]
ax=sns.clustermap(data=df_,cmap='RdBu',vmax=1,center=0,
                  row_cluster=False,col_cluster=False,
                  row_colors=colors, col_colors=colors,square=True)
# ax=sns.clustermap(data=df_overlap,cmap='Blues',vmin=0,vmax=1,
#                   row_colors=colors, col_colors=colors,square=True)
plt.savefig(os.path.join(logdir_,'spearmanr_all.svg'))
plt.close()

categories = [method2category[m] for m in df_.index]
subset_categories = np.unique(categories)
stats_comparisons = []
for category in subset_categories:
    idx = np.where(np.isin(categories,category))[0]
    stats_list = []
    for mat_ in [coefs,pvals]:
        mat_ = mat_[idx][:,idx]
        if True:
            mat_ = np.triu(mat_).reshape(-1)
            mat_ = mat_[mat_ != 0]
        else:
            mat_ = mat_.reshape(-1)
        stats_list.append(mat_.mean())
        stats_list.append(np.median(mat_))
    stats_comparisons.append(stats_list)

df_stats = pd.DataFrame(stats_comparisons,
                        columns = ['Coef Mean','Coef Median', 'Pval Mean', 'Pval Median'],
                        index = subset_categories)

#### Subset the methods
subset_plot = ['siVAE','siVAE-0','siVAE-l2-GRNBOOST2','LDVAE',
               'gene_relevance_input-ARACNE',
               'gene_relevance_input-MRNET',
               'gene_relevance_input-CLR',
               'gene_relevance_input-GRNBOOST2']

df_ = df_.reindex(subset_plot).transpose().reindex(subset_plot)
colors = [colors_dict[method2category[m]] for m in df_.index]
colors = [sns.color_palette()[0]] * 5 + [sns.color_palette()[1]] * 4

sns.set(font_scale=2)
colors = [colors_dict[method2category[m]] for m in df_.index]
ax=sns.clustermap(data=df_,cmap='RdBu',vmax=1,center=0,
                  row_cluster=False,col_cluster=False,
                  row_colors=colors, col_colors=colors,square=True)
# ax=sns.clustermap(data=df_overlap,cmap='Blues',vmin=0,vmax=1,
#                   row_colors=colors, col_colors=colors,square=True)
plt.savefig(os.path.join(logdir_,'spearmanr_subset.svg'))
plt.close()

#### Draw histogram + barplot --------------------------------------------------
df_dict = {}
df_plot['Recon Loss'] = df_plot['Test']
df_plot['Accuracy'] = (1 - df_plot['Test'])
bins = np.arange(10) * 0.1 + 0.1

## Wilcoxon test for difference

df_plot_ = df_plot.iloc[np.isin(df_plot.index,df_plot[df_plot.Method=='siVAE'].index)]
df_methods = {m: df_plot[df_plot.Method == m].reindex(scores_list[11].index) for m in pd.unique(df_plot['Method'])}

ax = sns.barplot(data=df_plot,x='Gene',y='Accuracy',zorder=0)
points_df=df_plot[np.isin(df_plot.Method,['NegLinear','Linear'])]
# sns.scatterplot(data=points_df, x='Gene',y='Accuracy',join=False, color='red', ax=ax, zorder=1, size=0.001)
sns.scatterplot(data=points_df, x='Gene',y='Accuracy',hue='Method', ax=ax, zorder=1, size=0.0001)
plt.legend([],[], frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(logdir_,'Accuracy_per_gene.svg'))
plt.close()

def calc_wilcoxon(x):
    if np.all(x == 0):
        return -1,-1
    else:
        return stats.wilcoxon(x)

## with wilcoxon
coefs = np.array([[calc_wilcoxon(df1.Accuracy - df2.Accuracy)[0] for _,df2 in df_methods.items()] for _,df1 in df_methods.items()])
pvals = np.array([[calc_wilcoxon(df1.Accuracy - df2.Accuracy)[1] for _,df2 in df_methods.items()] for _,df1 in df_methods.items()])

# ## with spearmanr
# coefs = np.array([[stats.spearmanr(df1.Accuracy,df2.Accuracy)[0] for _,df2 in df_methods.items()] for _,df1 in df_methods.items()])
# pvals = np.array([[stats.spearmanr(df1.Accuracy,df2.Accuracy)[1] for _,df2 in df_methods.items()] for _,df1 in df_methods.items()])

diffs = np.array([[(df1.Accuracy - df2.Accuracy).to_numpy() for _,df2 in df_methods.items()] for _,df1 in df_methods.items()])
mat_ = diffs[idx_not][:,idx_not]
mat_ = np.triu(mat_).reshape(-1)
mat_ = mat_[mat_ != 0]

idx_not_LDVAE  = np.where(np.array(list(grn_mat_dict.keys())) != 'LDVAE')[0]
idx_not_ARACNE = np.where(np.array(list(grn_mat_dict.keys())) != 'ARACNE')[0]
idx_not = np.intersect1d(idx_not_LDVAE,idx_not_ARACNE)
idx_dict = {'All_no_LDVAE': idx_not}

comparisons = []
comparisons.append(['All_no_LDVAE','All_no_LDVAE'])

stats_comparisons = []
for t1,t2 in comparisons:
    stats_list = []
    for mat_ in [coefs,pvals]:
        mat_ = mat_[np.array(idx_dict[t1])][:,np.array(idx_dict[t2])]
        if t1==t2:
            mat_ = np.triu(mat_).reshape(-1)
            mat_ = mat_[mat_ != 0]
        else:
            mat_ = mat_.reshape(-1)
        stats_list.append(mat_.mean())
        stats_list.append(np.median(mat_))
    stats_comparisons.append(stats_list)

comparisons_str = ["{} vs {}".format(c1,c2) for c1,c2 in comparisons]
df_stats = pd.DataFrame(stats_comparisons,
                        columns = ['Coef Mean','Coef Median', 'Pval Mean', 'Pval Median'],
                        index = comparisons_str)


idx_subset = np.where(np.isin(np.array(list(grn_mat_dict.keys())),subset_plot))[0]
method2idx = {key:ii for ii,key in enumerate(grn_mat_dict.keys())}
idx_subset = [method2idx[m] for m in subset_plot]
scores_list_subset = [scores_list[ii] for ii in idx_subset]
df_plot = pd.concat(scores_list_subset)
df_plot['Recon Loss'] = df_plot['Test']
df_plot['Accuracy'] = (1 - df_plot['Test'])
df_dict['Subset methods'] = df_plot

category = df_plot.Category.to_numpy()
category[category == 'VAE'] = 'siVAE'
df_plot.Category = category == 'VAE'

## All genes
sns.set_style(style='white')
for figure_name, df_plot in df_dict.items():
    logdir_plot = os.path.join(logdir_n_neighbors,'plot',figure_name)
    os.makedirs(logdir_plot,exist_ok=True)
    g = sns.FacetGrid(df_plot,col = 'Method')
    g.map(sns.distplot, 'Recon Loss', norm_hist=False, kde=False, bins=bins)
    plt.savefig(os.path.join(logdir_plot,'DistPlot_Recon_All.svg'))
    plt.close()
    ##
    g = sns.barplot(data=df_plot,x='Method',y='Recon Loss',hue='Category',dodge=False)
    plt.xticks(rotation=90)
    remove_spines(g,xaxis=False,show_legend=False)
    plt.tight_layout()
    plt.savefig(os.path.join(logdir_plot,'Bar_Recon_All.svg'))
    plt.close()
    ##
    g = sns.barplot(data=df_plot,x='Method',y='Accuracy',hue='Category',dodge=False)
    plt.xticks(rotation=90)
    remove_spines(g,xaxis=False,show_legend=False)
    plt.tight_layout()
    plt.savefig(os.path.join(logdir_plot,'Bar_Accuracy_All.svg'))
    plt.close()
    ##
    g = sns.barplot(data=df_plot,x='Method',y='Accuracy',hue='Category',dodge=False)
    plt.xticks(rotation=90)
    remove_spines(g,xaxis=False)
    leg = plt.legend(frameon=True,framealpha=1)
    plt.savefig(os.path.join(logdir_plot,'Bar_Accuracy_All_legend.svg'))
    plt.close()
    ##
    if figure_name == 'Subset methods':
        palette = sns.color_palette("Blues")[::-1][:4] + sns.color_palette("Oranges")[::-1][0:4]
        ##
        g = sns.barplot(data=df_plot,x='Method',y='Accuracy',dodge=False,palette=palette)
        plt.xticks(rotation=90)
        remove_spines(g,xaxis=False,show_legend=False)
        plt.tight_layout()
        plt.savefig(os.path.join(logdir_plot,'Bar_Accuracy_All.svg'))
        plt.close()
        ##
        g = sns.barplot(data=df_plot,x='Method',y='Accuracy',hue='Category',dodge=False,palette=palette)
        plt.xticks(rotation=90)
        remove_spines(g,xaxis=False)
        leg = plt.legend(frameon=True,framealpha=1)
        plt.savefig(os.path.join(logdir_plot,'Bar_Accuracy_All_legend.svg'))
        plt.close()

#### ---------------------------------------------------------------------------
## Moving average based on siVAE
genes_ranked_ = df_adj.sort_values('siVAE',ascending=False).index

df_list = [df_plot[df_plot.Method==m] for m in np.unique(df_plot.Method)]
df_list = [df.reindex(genes_ranked_) for df in df_list]
for df in df_list:
    df['rank'] = (np.arange(df.shape[0]) + 1)
    df['Recon Loss Average'] = df['Recon Loss'].cumsum() / (np.arange(df.shape[0]) + 1)

df_plot_ = pd.concat(df_list)
sns.lineplot(data=df_plot_,x='rank',y='Recon Loss Average',hue='Method')
plt.savefig(os.path.join(logdir_,'Lineplot_Recon_Loss_Average_siVAE_rank.pdf'))
plt.close()

bin_name = ['{}-{}'.format(df['rank'][0],df['rank'][-1]) for df in np.array_split(df_list[0],10)]
binned_mean = [[df2['Recon Loss'].mean() for df2 in np.array_split(df,10)] for df in df_list]
method_name = [np.unique(df.Method)[0] for df in df_list]
df_plot_ = pd.DataFrame({method:mean for mean,method in zip(binned_mean,method_name)},
                        index = bin_name)
df_plot_['Bin Range'] = df_plot_.index
df_plot_ = df_plot_.melt('Bin Range', var_name = 'Method', value_name = 'Binned Mean')
g = sns.lineplot(data=df_plot_,x='Bin Range',y='Binned Mean',hue='Method',sort=False)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(logdir_,'Lineplot_BinnedMean_siVAE_rank.pdf'))
plt.close()
