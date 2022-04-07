## System
import os
import gc
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

# Model
from siVAE.run_model import run_VAE
from siVAE.model.output import analysis

from siVAE import FeatureImportance as FI
from siVAE import classifier

from load_data import prepare_data

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import argparse

parser         = argparse.ArgumentParser()
method         = parser.add_argument('--method' , type=str, default='siVAE')
do_FA          = parser.add_argument('--do_FA'  , type=bool, default=False)
logdir         = parser.add_argument('--logdir' , type=str, default='out')
reduce_mode    = parser.add_argument('--reduce_mode',type=str, default='sample')
num_reduced    = parser.add_argument('--num_reduced',type=int, default=10000)
reduce_subset  = parser.add_argument('--reduce_subset',type=str, default="All")
datadir        = parser.add_argument('--datadir',type=str, default='default')
datadirbase    = parser.add_argument('--datadirbase',type=str, default='default')
use_full_data  = parser.add_argument('--use_full_data',type=bool, default=True)
dataset        = parser.add_argument('--dataset' , type=str, default='LargeBrain')
embedding_size = parser.add_argument('--embedding_size',type=int, default=20)
num_feature    = parser.add_argument('--num_feature',type=int, default=30000)
FA_method      = parser.add_argument('--FA_method' , type=str, default='All')

args = parser.parse_args()
method   = args.method
do_FA    = args.do_FA
logdir   = args.logdir
datadir  = args.datadir
dataset  = args.dataset
datadirbase = args.datadirbase
reduce_mode = args.reduce_mode
num_reduced = args.num_reduced
use_full_data = args.use_full_data
reduce_subset = args.reduce_subset
embedding_size = args.embedding_size
num_feature    = args.num_feature
FA_method      = args.FA_method

# ==============================================================================
#                           Specify Parameters for Model
# ==============================================================================

#### Set parameters ------------------------------------------------------------

#### Specify directory
os.makedirs(logdir,exist_ok=True)

#### Run Options
save_result = True
overwrite_result = False

## Specify feature attribution Methods to be used
if FA_method == 'All':
    method_DE = ['SaliencyMaps','GradInput','DeepLIFT',"IntGrad",'Shapley']
elif: FA_method == 'subset':
    method_DE = ['SaliencyMaps','GradInput','DeepLIFT']

#### Set up tf config
gpu_device = '0'
os.environ["CUDA_VISIBLE_DEVICES"]  = gpu_device
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
config.allow_soft_placement = True
config.intra_op_parallelism_threads = 5
config.inter_op_parallelism_threads = 5

#### Model Parameters
# iter       = 2000
iter       = 2000
mb_size    = 0.1
l2_scale   = 1e-3
dataAPI    = False
keep_prob  = 1
lr         = 2e-3
early_stop = False
dr         = 0.9

#### Additional settings
# Architecture should be a string with a specific format
# architecture: "Encoder-LE-Decoder-Output (0)-Index of LE"
architecture = '1024-512-128-LE-128-512-1024-0-3'
decoder_activation = 'NA'
zv_recon_scale = 0.05
decoder_var    = 'scalar'
do_pretrain    = True
LE_dim = 2

k_split = 0.8

#### Set parameters based on the model -----------------------------------------
if 'siVAE' in method:

    LE_method = 'siVAE'
    method_run = 'siVAE'
    raw = False
    output_distribution = 'normal'
    log_variational=False

    if 'linear' in method:
        architecture = architecture.split('LE')
        architecture = architecture[0]+'LE-0'+architecture[1].split('-0')[1]
        # lr= 2e-3

    if 'NB' in method:
        raw = True
        output_distribution = 'negativebinomial'
        log_variational=True

    if method == 'siVAE-0':
        zv_recon_scale=0

elif 'VAE' in method and not 'LDVAE':

    LE_method = 'VAE'
    method_run = 'siVAE'
    output_distribution = 'normal'
    raw = False
    log_variational=False

    if method == 'VAE-linear':

        raw = True
        output_distribution = 'negativebinomial'
        architecture = architecture.split('LE')
        architecture = architecture[0]+'LE-0'+architecture[1].split('-0')[1]
        log_variational=True

    elif method == 'VAE-NB':

        raw = True
        output_distribution = 'negativebinomial'

elif method in ['LDVAE','scVI']:

    raw = True
    method_run = 'scVI'
    encoder_architecture = architecture.split('-LE-')[0].split('-')
    n_layers = len(encoder_architecture)
    n_hidden_layers = [int(ii) for ii in encoder_architecture]

elif method == 'DegreeCentralityPrediction':

    raw = False
    architecture = architecture.split('LE')
    architecture = 'LE'+architecture[1].rsplit('-',1)[0]+'-0'
    # architecture = 'LE-128-512-1024-0-0'
    method_run = 'DegreeCentralityPrediction'

else:
    raise Exception('Input valid method type')



#### Load data --------------------------------------------------------------

## Use for full data
kwargs_FI, sample_set, datah_feature, datah_sample, plot_args = prepare_data(sample_size = 100,
                                                                             num_hvgs    = num_feature,
                                                                             raw         = True,
                                                                             reduce_mode = reduce_mode,
                                                                             num_reduced = num_reduced,
                                                                             reduce_subset = reduce_subset)
datah_sample.create_split_index_list(k_split=k_split,random_seed=0)


if raw:
    datah_sample.convert2raw()
    datah_feature.convert2raw()

del datah_sample.X.raw
del datah_feature.X.raw
gc.collect()


# ==============================================================================
#                                 Run siVAE
# ==============================================================================

if method_run == 'siVAE':

    ## Iterate over k-fold
    for k_fold in range(max(k_split,1)):

        logdir_out = os.path.join(logdir,'kfold-{}'.format(k_fold))
        siVAE_result_dir = os.path.join(logdir_out,'siVAE_result.pickle')

        datah_sample.create_dataset(kfold_idx=k_fold)

        if not overwrite_result and os.path.isfile(siVAE_result_dir):

            siVAE_result = util.load_pickle(siVAE_result_dir)
            plot_args['sample']['labels'] = siVAE_result.get_model().get_value('labels')[0]

        else:

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
                          'log_frequency': 50,
                          'learning_rate': lr,
                          "early_stopping"   : early_stop,
                          "validation_split" : 0,
                          "decay_rate"       : dr,
                          "decay_steps"      : 1000,
                          'var_dependency'   : True,
                          'activation_fun'   : tf.nn.relu,
                          'activation_fun_decoder': tf.nn.relu,
                          'output_distribution': output_distribution,
                          'beta'               : 1,
                          'l2_scale_final'     : 5e-3,
                          'log_variational'    : log_variational,
                          'beta_warmup'        : 1000,
                          'max_patience_count' : 100}

            graph_args['logdir_tf'] = logdir_out
            os.makedirs(logdir_out,exist_ok=True)
            ## Run model
            siVAE_result = run_VAE(graph_args_sample = graph_args,
                                  LE_method         = LE_method,
                                  datah_sample      = datah_sample,
                                  datah_feature     = datah_feature,
                                  zv_recon_scale    = zv_recon_scale,
                                  do_pretrain       = True,
                                  method_DE         = method_DE,
                                  sample_set        = sample_set,
                                  do_FA             = do_FA,
                                  kwargs_FI         = kwargs_FI)
            if save_result:
                util.save_pickle(siVAE_result, siVAE_result_dir)

                analysis.save_losses(siVAE_result,logdir_out)

            from siVAE.model.output.plot import plot_scalars
            logdir_plot = os.path.join(logdir,'plot')
            os.makedirs(logdir_plot,exist_ok=True)
            plot_scalars(siVAE_result, logdir_plot)

        # ==============================================================================
        #                               Run Analysis
        # ==============================================================================

        palette   = plot_args['palette']
        hue_order = plot_args['hue_order']

        methods_loadings = ['siVAE'] + method_DE
        gene_names = datah_sample.X.var_names.to_numpy()

        adata= datah_sample.X

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
            #### Plot
            df.columns = ['Dim 1', 'Dim 2', 'Label']
            df['Train-Test'] = np.isin(range(len(df)),siVAE_result.get_model().get_value('split_index')[0])
            ax = sns.scatterplot(data = df,
                            x = 'Dim 1',
                            y = 'Dim 2',
                            hue = 'Label',
                            style = 'Train-Test',
                            s = 3,
                            edgecolor=None,
                            hue_order = hue_order,
                            palette = palette)
            ax.legend_.remove()
            plt.savefig(os.path.join(logdir_out,'Latent_{}.pdf'.format(model_type)))
            plt.close()

        df_clf_accuracy = pd.DataFrame(results_clf_dict)
        df_clf_accuracy.to_csv(os.path.join(logdir_out,'clf_accuracy.csv'),index=False)
