## Load result for sample-reduce experiment

import os

## Maths
import math
import numpy as np
import pandas as pd

import logging
logging.getLogger('tensorflow').disabled = True
logging.getLogger().setLevel(logging.INFO)

## Plots
import matplotlib
# matplotlib.use('Agg') # Use for clusters
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

## siVAE
from siVAE import util
from siVAE import classifier

from sklearn.model_selection import StratifiedKFold

logdir = 'out/siVAE_reduce7'
logdir_list = os.listdir(logdir)
logdir_list = [i for i in logdir_list if '.svg' not in i]
logdir_list = [i for i in logdir_list if '.csv' not in i]
logdir_list.sort()

def load_clf(logdir_out):
    clf_accuracy = pd.read_csv(os.path.join(logdir_out,'clf_accuracy.csv'))
    clf_accuracy = clf_accuracy.melt()
    clf_accuracy.columns = ['Model', 'Accuracy']
    clf_accuracy['Reduce']        = logdir_model
    clf_accuracy['Reduce mode']   = logdir_model.split('-')[0]
    clf_accuracy['Reduce number'] = int(logdir_model.split('-')[1])
    return clf_accuracy

def load_losses(logdir_out):
    """
    """
    type2csv = {'model'  :'losses.csv',
                'sample' :'losses_sample.csv',
                'feature':'losses_feature.csv'}
    df_concat=[]
    for model in ['model','sample','feature']:
        losses = pd.read_csv(os.path.join(logdir_out,type2csv[model]))
        losses.index = losses.name
        if model == 'model':
            subset_loss  = ['total loss','decoder_loss','recon_loss','zv recon loss']
            losses = losses.reindex(subset_loss)
        else:
            subset_loss  = ['Total loss','KL_loss','recon loss']
            losses = losses.reindex(subset_loss)
            losses.index = ['total loss', 'decoder_loss', 'recon_loss']
        losses['Model'] = model
        df_concat.append(losses)
    df_comb = pd.concat(df_concat)
    df_comb['Name'] = df_comb.index
    return df_comb


def annotate_genes(df_plot_,gs_dict_in, discard_none=True):
    df_plot_['Geneset'] = 'None'
    for gs_,genes in gs_dict_in.items():
        df_plot_['Geneset'].loc[np.isin(df_plot_.index,genes)] = gs_
    if discard_none:
        df_plot_ = df_plot_[df_plot_.Geneset != 'None']
    df_plot_['Geneset_order'] = df_plot_['Geneset'] != 'None'
    return df_plot_


#### Load the datas
df_clf = []
df_losses = []
for logdir_model in logdir_list:
    logdir_out = os.path.join(logdir,logdir_model,'kfold-0')
    ## Load classifier accuracy
    df_clf.append(load_clf(logdir_out))
    ## Load losses
    df_comb = load_losses(logdir_out)
    df_comb['Reduce']        = logdir_model
    df_comb['Reduce mode']   = logdir_model.split('-')[0]
    df_comb['Reduce number'] = logdir_model.split('-')[1]
    df_losses.append(df_comb)

df_clf    = pd.concat(df_clf)
df_clf    = df_clf.sort_values(['Reduce mode','Reduce number'])
df_losses = pd.concat(df_losses)

#### Plot classification accuracy
for model in np.unique(df_clf.Model):
    df_plot = df_clf[df_clf.Model==model]
    ax = sns.barplot(data=df_plot,hue='Reduce mode',x='Reduce',y='Accuracy', dodge=False)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(logdir,"clf-{}.svg".format(model)))
    plt.close()

#### Plot training losses
df_plot = df_losses
g = sns.FacetGrid(data=df_plot,col='Model',row='Name',
                  sharex=False,sharey=False)
g.map(sns.barplot,'Reduce','test')
g.set_xticklabels(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(logdir,"losses.svg"))
plt.close()

# ------------------------------------------------------------------------------
# Plot feature embeddings with annotations
# ------------------------------------------------------------------------------

import gseapy as gp
from matplotlib.backends.backend_pdf import PdfPages
from siVAE.util import remove_spines

#### Set custom grouping of genesets -------------------------------------------

genesets = {"scsig":"/home/yongin/projects/siVAE/data/MSigDB/scsig.all.v1.0.1.symbols.gmt",
            "Hallmark":"/home/yongin/projects/siVAE/data/MSigDB/h.all.v7.1.symbols.gmt",
            "KEGG":"/home/yongin/projects/siVAE/data/MSigDB/c2.cp.kegg.v7.1.symbols.gmt"}

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

#### ---------------------------------------------------------------------------
#### Create data frame where each rows are genes and columns are metadata/stat
results_clf_dict = {}

for exp_name in logdir_list:
    logdir_exp = os.path.join(logdir,exp_name,'kfold-0')
    logdir_siVAE_result = os.path.join(logdir_exp,'siVAE_result.pickle')
    siVAE_result = util.load_pickle(logdir_siVAE_result)
    #
    logdir_gsea = os.path.join(logdir_exp,'GSEA')
    os.makedirs(logdir_gsea,exist_ok=True)
    #
    gene_embeddings = siVAE_result.get_feature_embeddings()
    gene_names = siVAE_result.get_model().get_value('var_names')
    X_plot, dim_labels = util.reduce_dimensions(gene_embeddings, reduced_dimension = 2, method = 'tSNE')
    df_plot = pd.concat([pd.DataFrame(X_plot), pd.DataFrame(gene_names)], axis = 1)
    df_plot.columns = dim_labels + ["Label"]
    df_plot.index = gene_names
    recons = siVAE_result.get_model().get_value('reconstruction')
    recon_loss_per_gene = np.square(recons[0] - recons[1]).mean(0)
    df_plot['Recon Loss per gene'] = recon_loss_per_gene
    #
    ## Plot setting
    plt.rcParams['patch.linewidth'] = 0
    plt.rcParams['patch.edgecolor'] = 'none'
    plt.rcParams["patch.force_edgecolor"] = False
    plt.rcParams['scatter.edgecolors'] = "none"
    #
    ## Plot annotated genese
    df_plot_ = df_plot.copy()
    df_plot_ = df_plot_.sort_values('Recon Loss per gene',ascending=True)[:500]
    fig_types = ['combined','individual','combined_excl','combined_subset']
    gs_dicts  = [gs_dict_new,gs_dict,gs_dict_excl,gs_dict_comb_subset]
    name2gs   = {type:gs_dict for type,gs_dict in zip(fig_types,gs_dicts)}
    name2gs   = {'combined_excl': name2gs['combined_excl']}
    #
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
    #### Plot feature embedding overlayed with recon loss
    ax = sns.scatterplot(x = dim_labels[0], y = dim_labels[1],
                         hue = 'Recon Loss per gene', data = df_plot,
                         edgecolor='black',linewidth=0.2,s=150)
    remove_spines(ax)
    plt.savefig(os.path.join(logdir_exp,'FeatureEmbeddings-ReconLoss.svg'))
    plt.close()
    ## Perform classifier with 5-fold split
    df_plot_ = annotate_genes(df_plot,name2gs['combined_excl'])
    X_out = np.array([df_plot_['Dim 1'],df_plot_['Dim 2']]).transpose()
    y     = df_plot_.Geneset
    skf = StratifiedKFold(n_splits=5)
    skf.get_n_splits(X_out, y)
    results_clf = []
    for train_index, test_index in skf.split(X_out, y):
        X_train = X_out[train_index]
        X_test = X_out[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        n_neighbors = len(np.unique(y)) * 2
        result = classifier.run_classifier(X_train = X_train, X_test = X_test,
                                  y_train = y_train, y_test = y_test,
                                  classifier = "KNeighborsClassifier", max_dim = 1e6,
                                  args = {'n_neighbors':n_neighbors})[0]
        results_clf.append(result)
    results_clf_dict[exp_name] = np.array(results_clf)
    #### For Feature wise enc-dec ----------------------------------------------
    gene_embeddings = siVAE_result.get_value('feature').get_value('z_mu')
    gene_names = siVAE_result.get_model().get_value('var_names')
    X_plot, dim_labels = util.reduce_dimensions(gene_embeddings, reduced_dimension = 2, method = 'tSNE')
    df_plot = pd.concat([pd.DataFrame(X_plot), pd.DataFrame(gene_names)], axis = 1)
    df_plot.columns = dim_labels + ["Label"]
    df_plot.index = gene_names
    df_plot['Recon Loss per gene siVAE'] = recon_loss_per_gene
    recons = siVAE_result.get_value('feature').get_value('reconstruction')
    recon_loss_per_gene = np.square(recons[0] - recons[1]).mean(1)
    df_plot['Recon Loss per gene'] = recon_loss_per_gene
    ax = sns.scatterplot(x = dim_labels[0], y = dim_labels[1],
                         hue = 'Recon Loss per gene', data = df_plot,
                         edgecolor='black',linewidth=0.2,s=150)
    remove_spines(ax)
    plt.savefig(os.path.join(logdir_exp,'FeatureEmbeddings-ReconLoss-FW.svg'))
    plt.close()
    ax = sns.scatterplot(x = dim_labels[0], y = dim_labels[1],
                         hue = 'Recon Loss per gene siVAE', data = df_plot,
                         edgecolor='black',linewidth=0.2,s=150)
    remove_spines(ax)
    plt.savefig(os.path.join(logdir_exp,'FeatureEmbeddings-ReconLoss-FW-siVAE.svg'))
    plt.close()
    #### Feature
    df_plot_ = annotate_genes(df_plot,name2gs['combined_excl'])
    X_out = np.array([df_plot_['Dim 1'],df_plot_['Dim 2']]).transpose()
    y     = df_plot_.Geneset
    skf = StratifiedKFold(n_splits=5)
    skf.get_n_splits(X_out, y)
    results_clf = []
    for train_index, test_index in skf.split(X_out, y):
        X_train = X_out[train_index]
        X_test = X_out[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        n_neighbors = len(np.unique(y)) * 2
        result = classifier.run_classifier(X_train = X_train, X_test = X_test,
                                  y_train = y_train, y_test = y_test,
                                  classifier = "KNeighborsClassifier", max_dim = 1e6,
                                  args = {'n_neighbors':n_neighbors})[0]
        results_clf.append(result)
    results_clf_dict[exp_name+"-FWED"] = np.array(results_clf)


#### Feature clustering accuracy
df_clf_accuracy = pd.DataFrame(results_clf_dict)
df_clf_accuracy.to_csv(os.path.join(logdir,'clf_accuracy.csv'),index=False)
df_clf_gene = df_clf_accuracy.melt()
df_clf_gene.columns = ['Name','Accuracy']
df_clf_gene['Mode'] = [c.split('-')[0] for c in df_clf_gene['Name']]
df_clf_gene['Num']  = [int(c.split('-')[1]) for c in df_clf_gene['Name']]
df_clf_gene = df_clf_gene.sort_values(['Mode','Num'])
df_clf_gene['FWED'] = np.array(['FWED' in name for name in df_clf_gene.Name])
df_clf_gene['Name2'] = np.array([name.split('-FWED')[0] for name in df_clf_gene.Name])

ax = sns.barplot(data=df_clf_gene, hue='Mode', x='Name', y='Accuracy', dodge=False)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(logdir,"clf-{}-all.svg".format('Feature')))
plt.close()

## hue based on FWED
ax = sns.barplot(data=df_clf_gene, hue='FWED', x='Name2', y='Accuracy', dodge=True)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(logdir,"clf-{}-all-FWED.svg".format('Feature')))
plt.close()


## Subset to non-FWED
df_clf_gene = df_clf_gene[df_clf_gene == False]
ax = sns.barplot(data=df_clf_gene, hue='Mode', x='Name', y='Accuracy', dodge=False)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(logdir,"clf-{}.svg".format('Feature')))
plt.close()

import gc

# #### Feature clustering with PCA (not finished)
# for exp_name in logdir_list:
#     if len(exp_name.split("-")) == 3:
#         logdir_exp = os.path.join(logdir,exp_name,'kfold-0')
#         # logdir_siVAE_result = os.path.join(logdir_exp,'siVAE_result.pickle')
#         # siVAE_result = util.load_pickle(logdir_siVAE_result)
#         print(exp_name)
#         logdir_data = os.path.join(logdir,exp_name,'data_dict.pickle')
#         datah_dict = util.load_pickle(logdir_data)
#         adata = datah_dict['feature'].X
#         np.unique(adata.var['GSEA label'])
#         #
#         logdir_gsea = os.path.join(logdir_exp,'GSEA')
#         os.makedirs(logdir_gsea,exist_ok=True)
#         #
#         gene_embeddings = siVAE_result.get_feature_embeddings()
#         gene_names = siVAE_result.get_model().get_value('var_names')
#         X_plot, dim_labels = util.reduce_dimensions(gene_embeddings, reduced_dimension = 2, method = 'tSNE')
#         df_plot = pd.concat([pd.DataFrame(X_plot), pd.DataFrame(gene_names)], axis = 1)
#         df_plot.columns = dim_labels + ["Label"]
#         df_plot.index = gene_names
#         recons = siVAE_result.get_model().get_value('reconstruction')
#         recon_loss_per_gene = np.square(recons[0] - recons[1]).mean(0)
#         df_plot['Recon Loss per gene'] = recon_loss_per_gene
#     df_plot_ = annotate_genes(df_plot,name2gs['combined_excl'])
#     X_out = np.array([df_plot_['Dim 1'],df_plot_['Dim 2']]).transpose()
#     y     = df_plot_.Geneset
#     skf = StratifiedKFold(n_splits=5)
#     skf.get_n_splits(X_out, y)
#     results_clf = []
#     for train_index, test_index in skf.split(X_out, y):
#         X_train = X_out[train_index]
#         X_test = X_out[test_index]
#         y_train = y[train_index]
#         y_test = y[test_index]
#         n_neighbors = len(np.unique(y)) * 2
#         result = classifier.run_classifier(X_train = X_train, X_test = X_test,
#                                   y_train = y_train, y_test = y_test,
#                                   classifier = "KNeighborsClassifier", max_dim = 1e6,
#                                   args = {'n_neighbors':n_neighbors})[0]
#         results_clf.append(result)
#     results_clf_dict[exp_name] = np.array(results_clf)




#### Cell embeddings Scatterplot with Geneset Annotations ----------------------

#### need change
# cell_embeddings = siVAE_result.get_cell_embeddings()
# cell_labels = siVAE_result.get_model().get_value('labels')
# X_plot, dim_labels = reduce_dimensions(cell_embeddings, reduced_dimension = 2, method = 'tSNE')
# df_plot = pd.concat([pd.DataFrame(X_plot), pd.DataFrame(cell_labels)], axis = 1)
# df_plot.columns = dim_labels + ["Label"]
# df_plot.index = cell_labels
#
# ## Set cell type groupings
# cell_labels_dict = {k:k for k in np.unique(cell_labels)}
# cell_labels_dict['Hepatocyte'] = 'Hepatocytes'
# cell_labels_dict['Kupffer Cell'] = 'Kupffer_Cells'
# cell_labels_dict['NK'] = 'NK_NKT_cells'
# cell_labels_dict['Mono-NK'] = 'NK_NKT_cells'
# cell_labels_dict['Mac NK'] = 'NK_NKT_cells'
# # cell_labels_dict['Fibroblast'] = 'Stellate_cells'
# # cell_labels_dict['HSC/MPP'] = 'Stellate_cells'
# cell_labels_dict['pro B cell'] = 'MHC_II_pos_B'
# cell_labels_dict['pre B cell'] = 'MHC_II_pos_B'
# cell_labels_dict['pre pro B cell'] = 'MHC_II_pos_B'
# cell_labels_2 = [cell_labels_dict[c] for c in cell_labels]
#
# df_plot['Label'] = cell_labels_2
# df_plot_ = df_plot
# df_plot_ = df_plot_[np.isin(df_plot_['Label'],selected_ct)]
# df_plot_['Label'] = df_plot_['Label'].astype('category')
# df_plot_['Label'].cat.set_categories(selected_ct,inplace=True)
# df_plot_.sort_values('Label',inplace=True)
# ax = sns.scatterplot(x = dim_labels[0], y = dim_labels[1], hue = 'Label', data = df_plot_,
#                      edgecolor='black',linewidth=0.2)
# remove_spines(ax)
# ax.legend_.remove()
# plt.savefig(os.path.join(logdir_gsea,'cell_embeddings_subset.svg'))
# plt.close()
