#### Scatterplot with Geneset Annotations --------------------------------------
## Need gene_embeddings/gene_names/df_single

import gseapy as gp

logdir_gsea = os.path.join(logdir,'GSEA')
os.makedirs(logdir_gsea,exist_ok=True)

genesets = {"scsig":"/home/yongin/projects/siVAE/data/MSigDB/scsig.all.v1.0.1.symbols.gmt",
            "Hallmark":"/home/yongin/projects/siVAE/data/MSigDB/h.all.v7.1.symbols.gmt",
            "KEGG":"/home/yongin/projects/siVAE/data/MSigDB/c2.cp.kegg.v7.1.symbols.gmt"}

#### Create data frame where each rows are genes and columns are metadata/stat
X_plot, dim_labels = reduce_dimensions(gene_embeddings, reduced_dimension = 2, method = 'tSNE')
df_plot = pd.concat([pd.DataFrame(X_plot), pd.DataFrame(gene_names)], axis = 1)
df_plot.columns = dim_labels + ["Label"]
df_plot.index = gene_names
df_plot['Recon Loss per gene'] = recon_loss_per_gene
df_plot['Distance from Origin'] = np.square(gene_embeddings).sum(1)

df_plot = pd.concat([df_plot,df_single],axis=1)

#### Set custom grouping of genesets
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

cell_embeddings = values_dict['sample']['z_mu']
cell_labels = result_dict['model']['labels']
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
