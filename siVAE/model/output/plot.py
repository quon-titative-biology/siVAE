import os

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

from siVAE.util import remove_spines
from siVAE.util import reduce_dimensions

def plot_scalars(siVAE_output, logdir):
    """ """
    result_model = siVAE_output.get_model()
    result_model.convert_scalars_to_df()
    df_tb = result_model.get_value('scalars_df')
    #
    logdir_plot_scalars = os.path.join(logdir,'scalars')
    os.makedirs(logdir_plot_scalars,exist_ok=True)
    for model in df_tb.Model.unique():
        df_model = df_tb[df_tb.Model == model]
        for attr in df_model.Name.unique():
            df_attr = df_model[df_model.Name == attr].drop_duplicates(['Step','Type'])
            df_attr.reset_index(drop=True,inplace=True)
            ax = sns.lineplot(data=df_attr,x='Step',y='Value',hue='Type')
            remove_spines(ax)
            ax.set(ylabel=attr)
            plt.savefig(os.path.join(logdir_plot_scalars,'{}-{}.svg'.format(model,attr)))
            plt.close()


def plot_gene_embeddings(siVAE_output, logdir, filename=None, hue=None, method_dim='PCA',**kwargs):
    #### For Dim
    X_in = siVAE_output.get_model().get_value('latent_embedding')['feature']
    X_out, dim_labels = reduce_dimensions(X_in, reduced_dimension = 2, method = method_dim)
    if hue is None:
        y = siVAE_output.get_model().get_value('var_names')
    #### Plot
    labels_test = y
    df_plot = pd.concat([pd.DataFrame(X_out), pd.DataFrame(labels_test)], axis = 1)
    df_plot.columns = dim_labels + ["Label"]
    ax = sns.scatterplot(x = dim_labels[0], y = dim_labels[1], hue = "Label", data = df_plot, **kwargs)
    remove_spines(ax)
    ax.legend_.remove()
    if filename is None:
        filename = 'LatentEmbedding.svg'
    plt.savefig(os.path.join(logdir,filename))
    plt.close()


def plot_latent_embeddings(siVAE_output, logdir, type = 'Sample', filename=None,
                           method_dim='PCA', hue=None, show_legend=True, **kwargs):
    #### For Dim
    if type == 'Sample':
        X_in = siVAE_output.get_model().get_sample_embeddings()
    elif type == 'Feature':
        X_in = siVAE_output.get_model().get_feature_embeddings()
    else:
        raise Exception('Input valid type (Sample or Feature)')
    X_out, dim_labels = reduce_dimensions(X_in,
                                          reduced_dimension = 2,
                                          method = method_dim)
    if hue is None:
        if type == 'Sample':
            y = siVAE_output.get_model().get_value('labels')
        elif type == 'Feature':
            y = siVAE_output.get_model().get_value('var_names')
        else:
            raise Exception('Input valid type [sample, feature]')
    else:
        y = hue
    #### Plot
    df_plot = pd.concat([pd.DataFrame(X_out), pd.DataFrame(y)], axis = 1)
    df_plot.columns = dim_labels + ["Label"]
    ax = sns.scatterplot(x = dim_labels[0], y = dim_labels[1], hue = "Label", data = df_plot, **kwargs)
    remove_spines(ax)
    if not show_legend:
        ax.legend_.remove()
    if filename is None:
        filename = '{}Embedding.svg'.format(type)
    plt.savefig(os.path.join(logdir,filename))
    plt.close()
