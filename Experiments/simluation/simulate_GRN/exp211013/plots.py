## Plots
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

import anndata

def plot_expressions_multiple(adata=None, X=None, obs=None, var=None,
                              logdir="", plot_tf=True,plot_target=True,**kwargs):

    # Create adata from components if not directly inputted
    if adata is None:
        adata = anndata.AnnData(X,obs=obs,var=var)
    else:
        adata = adata.copy()

    # Plot per group
    for ii,g in enumerate(adata.var.Group.unique()):

        # Plot all genes in a group
        logdir1, ext = logdir.rsplit('.',1)
        filename = "{}-{}.{}".format(logdir1,g,ext)
        adata_single = adata[:,adata.var.Group == g]
        adata_single.obs = pd.DataFrame(adata.obs.State).iloc[:,[ii]]
        _ = plot_expressions(adata_single, **kwargs)
        plt.savefig(filename)
        plt.close()

        # Plot only TF in a group
        logdir1, ext = logdir.rsplit('.',1)
        filename = "{}-{}-TF.{}".format(logdir1,g,ext)
        adata_single = adata_single[:,adata_single.var.Type == 'TF']
        _ = plot_expressions(adata_single, **kwargs)
        plt.savefig(filename)
        plt.close()

    # Plot only the tfs
    if plot_tf:
        # rename file
        logdir1, ext = logdir.rsplit('.',1)
        filename = "{}-{}.{}".format(logdir1,'TF',ext)
        adata_single = adata[:,adata.var.Type == 'TF']
        g = plot_expressions(adata_single, **kwargs)
        plt.savefig(filename)
        plt.close()

    # Plot stats for targets
    if plot_target:

        logdir1, ext = logdir.rsplit('.',1)
        filename = "{}-{}.{}".format(logdir1,'target',ext)

        adata_single = adata[:,adata.var.Type == 'target']

        # Calculate mean/var expressions per group for each cell
        d = []
        metrics = ['Mean abs', 'Mean', 'Var']
        ids     = ['Group', 'State']
        for ii,g in enumerate(adata_single.var.Group.unique()):

            adata_group = adata_single[:,adata_single.var.Group == g]
            states = pd.DataFrame(adata_group.obs.State).iloc[:,[ii]] # extract state of gene group ii
            states = states.to_numpy().reshape(-1)

            for s in np.unique(states):

                adata_state = adata_group[states == s]
                abs_mean = np.abs(adata_state.X).mean(-1).mean()
                reg_mean = adata_state.X.mean(-1).mean()
                var_mean = adata_state.X.var(-1).mean()
                d.append((abs_mean,reg_mean,var_mean,g,s))

        df_plot = pd.DataFrame(d,columns = metrics + ids)
        df_plot = df_plot.melt(id_vars    = ids,
                               value_vars = metrics,
                               var_name   = 'Metric',
                               value_name = 'Value')
        df_plot.Value = df_plot.Value.astype('float32')
        sns.catplot(data=df_plot, x='Group', y='Value',
                    hue='State', col='Metric', kind = 'bar')
        plt.savefig(filename)
        plt.close()


def plot_expressions(adata=None, X=None,
                     obs=None, var=None,
                     n_sample=200, row_cluster=False,
                     row='cell', col='gene',center=0,cmap='RdBu',**kwargs):
    """ Plot expression as heatmap with annotations """

    if adata is not None:
        X = adata.X
        obs = adata.obs
        var = adata.var

    if n_sample is not None:
        if isinstance(n_sample,int):
            n_sample = np.arange(n_sample)
        n_sample = np.array(n_sample)
        X_plot = X[n_sample]
        obs    = obs.iloc[n_sample]

    else:
        X_plot = X

    def map2color(series,palette=None,lut=None):
        """ Map series 2 cmap """
        if lut is None:
            lut = dict(zip(series.unique(),palette))
        return series.map(lut)

    def df2color(df, type = 'gene'):

      if type == 'gene':
          col_colors1 = map2color(df.Group,sns.color_palette())
          col_colors2 = map2color(df.Type,sns.color_palette("Set2"))
          colors = pd.DataFrame({"Group": col_colors1, "Type": col_colors2})

      elif type == 'cell':
          row_colors = pd.DataFrame(df.State).apply(map2color,axis=0,lut={"Off":'grey',"On":'red'})
          row_colors.columns = [f"G{ii+1}" for ii in range(row_colors.shape[1])]
          colors = row_colors.reset_index(drop=True)

      else:
          colors=None

      return colors

    col_colors = df2color(var,col)
    row_colors = df2color(obs,row)

    g = sns.clustermap(X_plot,center=center,cmap=cmap,
                      row_cluster=row_cluster, col_cluster=False,
                      col_colors=[series for _,series in col_colors.iteritems()],
                      row_colors=[series for _,series in row_colors.iteritems()],
                      **kwargs)

    g.ax_row_dendrogram.set_visible(False)
    ax = g.ax_heatmap
    ax.set_xlabel(col)
    ax.set_ylabel(row)
    plt.tight_layout()
    return g

def plot_scatter_states(X_dict, states, filename):
    """  """
    n_rows = states.shape[-1]
    n_cols = len(X_dict)
    fig, axs = plt.subplots(n_rows, n_cols,
                            squeeze=False,
                            figsize=(n_cols * 5, n_rows * 4)
                            )
    ## col vs row
    for ii, (method, X_loadings) in enumerate(X_dict.items()):
        print(method)
        df_plot = pd.DataFrame(X_loadings, columns=['Dim 1','Dim 2'])
        df_plot = pd.concat([df_plot,states],axis=1)
        for jj in range(n_rows):
            state_n = states.columns[jj]
            ax = axs[jj][ii]
            ax = sns.scatterplot(data=df_plot,x='Dim 1',y='Dim 2',
                                 edgecolor='black',linewidth=0.5,
                                 hue=state_n,ax=ax)
            xmax = df_plot['Dim 1'].abs().max() * 1.2
            ymax = df_plot['Dim 2'].abs().max() * 1.2
            _ = ax.set_xlim([-xmax,xmax])
            _ = ax.set_ylim([-ymax,ymax])
            _ = ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            _ = ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            legend = ax.get_legend()
            _ = legend.remove()
            ax.set_title(f"State-{state_n}-{method}")
            #
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(filename)
    #
    ## plot legends
    fig = plt.figure()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels)
    plt.tight_layout()
    plt.savefig(filename.rsplit(".",1)[0]+"_legend"+filename.rsplit(".",1)[1])
    plt.close()


def plot_scatter_genes(X_dict, var, logdir, center=True):
    """
    center: True then center to origin
    """
    n_col = 3
    n_row = int(len(X_dict) / 3) + 1
    if n_row == 1:
        n_col = len(X_dict)
    fig, axes = plt.subplots(n_row, n_col, figsize=(11,7),squeeze=False)
    axes = axes.flatten()
    var['Size'] = var['Type'].map({'TF':50,'target':20})
    #
    for ii,(method,X_loadings) in enumerate(X_dict.items()):
        ax = axes[ii]
        df_plot = pd.DataFrame(X_loadings,columns=['Dim 1','Dim 2'])
        df_plot = pd.concat([df_plot,var.reset_index(drop=True)],axis=1)
        ax = sns.scatterplot(data=df_plot,x='Dim 1',y='Dim 2',ax=ax,
                             edgecolor='black',linewidth=0.5,
                             hue='Group',style='Type', size='Type', sizes=(200,400),
                             markers = ['X','o'])
        if center:
            xmax = df_plot['Dim 1'].abs().max() * 1.2
            ymax = df_plot['Dim 2'].abs().max() * 1.2
            _ = ax.set_xlim([-xmax,xmax])
            _ = ax.set_ylim([-ymax,ymax])
        _ = ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        _ = ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        legend = ax.get_legend()
        _ = legend.remove()
        ax.set_title(method)
    #
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, f'gene_embeddings.svg'))
    plt.close()
    #
    fig = plt.figure()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels)
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, f'gene_embeddings_legend.svg'))
    plt.close()


# def plot_losses(args_dict, logdir_VAE, do_plot=True, **kwargs):
#     # Plot comparisons of losses for models
#
#     df_list = []
#     for args in itertools.product(*args_dict.values()):
#         args_update = {k:v for k,v in zip(args_dict.keys(),args)}
#         args_s = args_update.copy()
#         args_str = os.path.join(*[f"{k}-{v}" for ii,(k,v) in enumerate(args_update.items())])
#         logdir_tf = os.path.join(logdir_VAE,args_str)
#         beta_feature = args_update.pop('beta_feature')
#         graph_args.update(args_update)
#         model_name = "-".join([f"{k}-{v}" for ii,(k,v) in enumerate(args_update.items())])
#         os.makedirs(logdir_tf,exist_ok=True)
#         # Add metrics to be plotted
#         metrics_dict = {}
#         # Losses
#         df_losses = pd.read_csv(os.path.join(logdir_tf,'losses.csv'))
#         if args_update['LE_method'] == 'siVAE':
#             loss_name = 'recon_loss'
#         else:
#             loss_name = 'recon loss'
#         metrics_dict['losses_train'] = df_losses[df_losses.name==loss_name]['train']
#         metrics_dict['losses_test']  = df_losses[df_losses.name==loss_name]['test']
#         args_s.update(metrics_dict)
#         df_new = pd.DataFrame(args_s).reset_index(drop=True)
#         df_list.append(df_new)
#
#     for LE_dim in args_dict["LE_dim"]:
#         logdir_plot = os.path.join(logdir,'pca',f'LE_dim-{LE_dim}')
#         metrics_dict = {}
#         # Losses
#         df_losses = pd.read_csv(os.path.join(logdir_plot,'pca_recon_loss.csv'))
#         if args_update['LE_method'] == 'siVAE':
#             loss_name = 'recon_loss'
#         else:
#             loss_name = 'recon loss'
#         metrics_dict['losses_train'] = df_losses['train']
#         metrics_dict['losses_test']  = df_losses['test']
#         df_new = pd.DataFrame(metrics_dict).reset_index(drop=True)
#         df_new['LE_method'] = 'PCA'
#         df_new['LE_dim'] = LE_dim
#         df_list.append(df_new)
#
#     df_plot = pd.concat(df_list)
#
#         if do_plot:
#             for train_type in ['train','test']:
#                 sns.barplot(data=df_plot, x='LE_dim', y=f'losses_{train_type}', hue='LE_method', **kwargs)
#                 plt.savefig(os.path.join(logdir,f'Losses {train_type}.svg'))
#                 plt.close()
#
#         return df_plot
