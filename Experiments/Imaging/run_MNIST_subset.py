## System
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'  # no debugging from TF
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

import time
import copy

## sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

## Maths
import math
import numpy as np
import pandas as pd

import logging
logging.getLogger('tensorflow').disabled = True
logging.getLogger().setLevel(logging.INFO)

## Tensorflow
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

## Plots
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

import scanpy as sc

## siVAE
# Util
from siVAE import util
from siVAE.util import reduce_dimensions
from siVAE.util import remove_spines

# Model
from siVAE.run_model import run_VAE
from siVAE.model.output import analysis

from siVAE import FeatureImportance as FI
from siVAE import classifier

from load_MNIST import prepare_data

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import argparse

parser         = argparse.ArgumentParser()
method         = parser.add_argument('--method', type=str, default='siVAE')
do_FA          = parser.add_argument('--do_FA', type=bool, default=False)
logdir         = parser.add_argument('--logdir', type=str, default='out')

args = parser.parse_args()
method   = args.method
do_FA    = args.do_FA
logdir   = args.logdir

# ==============================================================================
#                           Specify Parameters for Model
# ==============================================================================

#### Run Options
save_result = True
overwrite_result = False

## Specify feature attribution Methods to be used
# method_DE = ['SaliencyMaps','GradInput','DeepLIFT',"IntGrad",'Shapley']
method_DE = ['SaliencyMaps','GradInput','DeepLIFT']

#### Set up tf config
gpu_device = '1'
os.environ["CUDA_VISIBLE_DEVICES"]  = gpu_device
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
config.allow_soft_placement = True
config.intra_op_parallelism_threads = 5
config.inter_op_parallelism_threads = 5

#### Model Parameters
iter       = 2000
mb_size    = 2000
l2_scale   = 5e-3
dataAPI    = False
keep_prob  = 1
lr         = 1e-3
early_stop = False
dr         = 0.8

#### Additional settings
# Architecture should be a string with a specific format
# architecture: "Encoder-LE-Decoder-Output (0)-Index of LE"
# architecture = '64-32-LE-32-64-0-2'
architecture = '512-128-32-LE-32-128-512-0-3'
decoder_activation = 'NA'
zv_recon_scale = 0.1
decoder_var    = 'scalar'
do_pretrain    = True
LE_dim = 2

graph_args = {'LE_dim'       : LE_dim,
              'architecture' : architecture,
              'config'       : config,
              'iter'         : iter,
              'mb_size'      : mb_size,
              'l2_scale'     : l2_scale,
              'dataAPI'      : dataAPI,
              'tensorboard'  : True,
              'batch_norm'   : False,
              'keep_prob'    : keep_prob,
              'log_frequency': 100,
              'learning_rate': lr,
              "early_stopping"   : early_stop,
              "validation_split" : 0,
              "decay_rate"       : dr,
              "decay_steps"      : 1000,
              'var_dependency'   : True,
              'activation_fun'   : tf.nn.relu}

graph_args['logdir_tf'] = logdir
os.makedirs(logdir,exist_ok=True)

LE_method = 'siVAE'

# ==============================================================================
#                                 Run Model
# ==============================================================================

## Save the train/test split for consistency and load
datadir = os.path.join('out/data_dict.pickle')
k_split=0.8
if os.path.exists(datadir):

    data_dict = util.load_pickle(datadir)

    datah_sample  = data_dict['sample']
    datah_feature = data_dict['feature']
    plot_args     = data_dict['plot_args']
    kwargs_FI     = data_dict['kwargs_FI']
    sample_set    = data_dict['sample_set']

else:
    ## Load and save data
    kwargs_FI, sample_set, datah_feature, datah_sample, plot_args = prepare_data(max_size=5000,
                                                                                 num_reduced=1000,
                                                                                 sample_size=1000,
                                                                                 subsets_labels=[1,6])

    datah_sample.create_split_index_list(k_split=k_split,random_seed=0)

    ## Save file
    data_dict = {'sample'    : datah_sample,
                 'feature'   : datah_feature,
                 'plot_args' : plot_args,
                 'kwargs_FI' : kwargs_FI,
                 'sample_set': sample_set}

    util.save_pickle(data_dict,datadir)

datah_sample.create_dataset(kfold_idx=0)

## Run model
siVAE_result = run_VAE(graph_args, LE_method, datah_sample,
                      zv_recon_scale=zv_recon_scale, datah_feature=datah_feature,
                      do_pretrain=True, method_DE=method_DE,
                      sample_set=sample_set, do_FA = do_FA,
                      kwargs_FI = kwargs_FI)

siVAE_result_dir = os.path.join(logdir,'siVAE_result.pickle')
if save_result:
    util.save_pickle(siVAE_result, siVAE_result_dir)

#### Plot training history
from siVAE.model.output.plot import plot_scalars

logdir_plot = os.path.join(logdir,'plot')
os.makedirs(logdir_plot,exist_ok=True)

plot_scalars(siVAE_result, logdir_plot)


# ==============================================================================
#                           Compare cell embeddings visualizations
# ==============================================================================

#### Ony plot 10000 sample
X_raw = datah_sample.X.X[:10000]
X_siVAE = siVAE_result.get_sample_embeddings()[:10000]
labels = datah_sample.X.obs[:10000]
for method in ['PCA','tSNE','UMAP','siVAE']:
    if method == 'siVAE':
        X_plot, dim_labels = reduce_dimensions(X_siVAE, reduced_dimension = 2, method = 'tSNE')
        dim_labels = ['siVAE-1','siVAE-2']
    else:
        X_plot, dim_labels = reduce_dimensions(X_raw, reduced_dimension = 2, method = method)
    df_plot = pd.DataFrame(X_plot)
    df_plot.columns = dim_labels
    df_plot['Label'] = labels.to_numpy()
    # Plot
    ax = sns.scatterplot(x = dim_labels[0], y = dim_labels[1],
                         hue = "Label", data = df_plot)
    remove_spines(ax)
    plt.savefig(os.path.join(logdir,'DR_vis-{}.svg'.format(method)))
    plt.close()


# ==============================================================================
#                               Run Analysis
# ==============================================================================

analysis.save_losses(siVAE_result,logdir)

from siVAE.model.output import plot
from siVAE.model.output.plot import plot_latent_embeddings

kwargs={'s':5,'edgecolor':"none"}
for type in ['Sample','Feature']:
    plot_latent_embeddings(siVAE_result,
                           logdir=logdir,
                           type=type,
                           filename='{}Embeddings.svg'.format(type),
                           show_legend=True,
                           method_dim='tSNE',
                           **kwargs)

# ==============================================================================
#                          Clustering accuracy
# ==============================================================================

logdir_out = logdir
#### Run classifier on the embedding spacemode
results_clf_dict = {}
for model_type in ['sample','model']:
    #### classifier
    ## Input
    if model_type == 'model':
        X_out = siVAE_result.get_model().get_sample_embeddings()
    elif model_type == 'sample':
        X_out = siVAE_result.get_value('sample').get_value('z_mu')
    y = siVAE_result.get_model().get_value('labels')
    df = pd.DataFrame(X_out)
    df['Labels'] = y
    df.to_csv(os.path.join(logdir_out,'latent_{}.csv'.format(model_type)))
    df = pd.read_csv(os.path.join(logdir_out,'latent_{}.csv'.format(model_type)),index_col=0)
    ## Run 5-fold classifier experiment
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
    results_clf_dict[model_type] = np.array(results_clf)

df_clf_accuracy = pd.DataFrame(results_clf_dict)
df_clf_accuracy.to_csv(os.path.join(logdir_out,'clf_accuracy.csv'),index=False)

# ==============================================================================
#                           Run Analysis for Image
# ==============================================================================

from siVAE.model.output.analysis import plot_siVAE_loadings
from siVAE.model.output.analysis import plot_FA_loadings

ImageDims = plot_args['ImageDims']

## Visualization for Image Datasets (ImageDims is not None)
## Plot loadings
ImageDims = [28,28,1]
plot_siVAE_loadings(siVAE_result, logdir, ImageDims)
plot_FA_loadings(siVAE_result, logdir, ImageDims)

## Save count data for gene relevnace loadings

datadir = 'out/data/siVAE'

if do_FA:
    n_sample = 1000
    gene_names = datah_sample.X.var_names.to_numpy()
    cell_names = datah_sample.X.obs['Labels'].to_numpy()[:n_sample]

    # Get count data
    adata_raw = datah_sample.X.raw
    idx_genes = np.array([np.where(adata_raw.var_names == ii)[0][0] for ii in gene_names])
    exprs  = adata_raw[:,idx_genes].X[:n_sample]
    coords = siVAE_result.get_sample_embeddings()[:n_sample]

    ## Save count data for gene relevance
    os.makedirs(datadir,exist_ok=True)
    np.savez(file = os.path.join(datadir,'gene_relevance_input.npz'),
             exprs  = exprs,
             coords = coords,
             gene_names = gene_names,
             cell_names = cell_names)
