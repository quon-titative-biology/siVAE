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

from repo_FetalLiver_2_SetupInput import prepare_data

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser         = argparse.ArgumentParser()
parser.add_argument('--method' , type=str, default='siVAE')
parser.add_argument('--do_FA'  , type=str2bool, default=False)
parser.add_argument('--logdir' , type=str, default='out')
parser.add_argument('--reduce_mode',type=str, default='sample')
parser.add_argument('--num_reduced',type=int, default=10000)
parser.add_argument('--reduce_subset',type=str, default="All")
parser.add_argument('--datadir',type=str, default='default')
parser.add_argument('--datadirbase',type=str, default='default')
parser.add_argument('--use_full_data',type=bool, default=True)
parser.add_argument('--use_batch',type=bool, default=False)

parser.add_argument('--LE_dim',type=int, default=2)
parser.add_argument('--k_split',type=float, default=0.8)

parser.add_argument('--mb_size',type=float, default=128)
parser.add_argument('--iter',type=int, default=2000)

parser.add_argument('--zv_recon_scale', type=float, default=None)

args = parser.parse_args()

# Directories
logdir   = args.logdir
datadir  = args.datadir
datadirbase = args.datadirbase

# Model set up
LE_dim = args.LE_dim
method   = args.method
do_FA    = args.do_FA

# Preprocessing steps
reduce_mode = args.reduce_mode
num_reduced = args.num_reduced
use_full_data = args.use_full_data
reduce_subset = args.reduce_subset

# Experimental setup
k_split = args.k_split

# Training setup
mb_size = args.mb_size
iter    = args.iter

# Model parameters
zv_recon_scale = args.zv_recon_scale
use_batch      = args.use_batch

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
method_DE = ['SaliencyMaps','GradInput']

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
l2_scale   = 1e-3
dataAPI    = False
keep_prob  = 1
lr         = 2e-3
early_stop = False
dr         = 0.9

# iter       = 2000
# mb_size    = 0.1
# l2_scale   = 1e-3
# dataAPI    = False
# keep_prob  = 1
# lr         = 2e-3
# early_stop = False
# dr         = 0.9

#### Additional settings
# Architecture should be a string with a specific format
# architecture: "Encoder-LE-Decoder-Output (0)-Index of LE"
architecture = '1024-512-128-LE-128-512-1024-0-3'
decoder_activation = 'NA'
decoder_var    = 'scalar'
do_pretrain    = True

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


## Save the train/test split for consistency and load
if datadir == 'default':
    datadir = os.path.join('out/data_dict.pickle')

if os.path.isfile(datadir):

    data_dict = util.load_pickle(datadir)

    datah_sample  = data_dict['sample']
    datah_feature = data_dict['feature']
    plot_args     = data_dict['plot_args']
    kwargs_FI     = data_dict['kwargs_FI']
    sample_set    = data_dict['sample_set']

else:

    if use_full_data:

        ## Use for full data
        kwargs_FI, sample_set, datah_feature, datah_sample, plot_args = prepare_data(sample_size = 100,
                                                                                     num_hvgs    = 2000,
                                                                                     raw         = True,
                                                                                     reduce_mode = reduce_mode,
                                                                                     num_reduced = num_reduced,
                                                                                     reduce_subset = reduce_subset)

    else:
        ## Use reduced size for testing
        subsets_labels = ['Hepatocyte','Kupffer Cell','NK','Mono-NK','Mac NK',
                          'pro B cell','pre B cell','pre pro B cell']
        kwargs_FI, sample_set, datah_feature, datah_sample, plot_args = prepare_data(sample_size    = 100,
                                                                                     num_sample     = 2000,
                                                                                     num_hvgs       = 500,
                                                                                     subsets_labels = subsets_labels,
                                                                                     raw            = True,
                                                                                     reduce_mode    = reduce_mode,
                                                                                     num_reduced    = num_reduced,
                                                                                     reduce_subset = reduce_subset)

    ## For consistent dataset split with reduce sample experiment
    if os.path.isfile(datadirbase):
        data_dict = util.load_pickle(datadirbase)
        datah_sample.split_index_list = data_dict['sample'].split_index_list
    else:
        datah_sample.create_split_index_list(k_split=k_split,random_seed=0)

    ## Save file
    data_dict = {'sample'    : datah_sample,
                 'feature'   : datah_feature,
                 'plot_args' : plot_args,
                 'kwargs_FI' : kwargs_FI,
                 'sample_set': sample_set}

    util.save_pickle(data_dict,datadir)

if raw:
    datah_sample.convert2raw()
    datah_feature.convert2raw()

# Convert2raw
del datah_sample.X.raw
del datah_feature.X.raw
gc.collect()

# ==============================================================================
#                        Run DegreeCentralityPrediction
# ==============================================================================

if method_run == 'DegreeCentralityPrediction':

    logdir_out = os.path.join(logdir)
    os.makedirs(logdir_out,exist_ok=True)

    from siVAE.model import regressor

    #### Model Parameters
    iter       = 1000
    mb_size    = 200
    l2_scale   = 5e-3
    dataAPI    = False
    keep_prob  = 0.9
    lr         = 1e-3
    early_stop = False
    dr         = 0.8

    #### Additional settings
    decoder_activation = 'NA'
    decoder_var    = 'scalar'
    do_pretrain    = True

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
                  'activation_fun'   : tf.nn.relu,
                  'l2_scale_final'   : 0,
                  'logdir_tf'        : logdir_out}

    ## Set up architecture of hidden layers
    LE_dim       = graph_args['LE_dim']
    architecture = graph_args['architecture']

    # For datah_sample
    h_dims = architecture.replace('LE',str(LE_dim))
    h_dims = [int(dim) for dim in h_dims.split("-")]
    datah_sample.index_latent_embedding = int(h_dims.pop())
    datah_sample.h_dims = h_dims

    results = []
    PVEs = []
    durations = []

    expr       = datah_sample.X.X
    gene_names = datah_sample.X.var_names.to_numpy()
    datah_reg  = datah_sample

    for ii in range(expr.shape[-1]):
        print(ii)
        start = time.time()
        datah_reg.X = datah_sample.X[:,0] # reduce size to sample x 1
        datah_reg.X.X = expr[:,[ii]]
        datah_reg.create_dataset(0)
        #
        reg_model = regressor.AutoEncoder(data_handler = datah_reg,
                                          **graph_args)
        reg_model.build_model()
        with tf.Session() as sess:
            result = reg_model.train(sess)
        ##
        result['reconstruction'][0]
        y      = result['reconstruction'][0]
        y_pred = result['reconstruction'][1]
        PVE = analysis.calculate_PVE(y,y_pred)
        recon_loss_per_gene = np.square(y - y_pred).mean(0)
        # np.save(os.path.join(logdir_out,'PVE-{}.npy'.format(ii)),PVE)
        # np.save(os.path.join(logdir_out,'recon_loss-{}.npy'.format(ii)),recon_loss_per_gene)
        results.append(recon_loss_per_gene)
        PVEs.append(PVE)
        #
        durations.append(time.time()-start)

    ## Save as npz
    np.save(os.path.join(logdir_out,'recon_loss-{}.npy'.format('all')),
            np.array(results))
    np.save(os.path.join(logdir_out,'PVE-{}.npy'.format('all')),
            np.array(PVEs))
    np.save(os.path.join(logdir_out,'gene_name.npy'),
            gene_names)
    np.savez(gene_names = datah_sample.X.var_names.to_numpy(),
             PVE        = np.array(PVEs),
             recon_loss = np.array(results),
             file       = os.path.join(logdir_out,'single_gene_exp.npz'))


# ==============================================================================
#                                 Run scVI
# ==============================================================================

if method_run == 'scVI':

    ## scvi-tools
    import scvi

    logdir_out = os.path.join(logdir)
    os.makedirs(logdir_out,exist_ok=True)

    lr       = 1e-3
    n_latent = LE_dim
    n_epochs = 100
    n_hidden = n_hidden_layers[-1]

    ## Change AnnData expression to raw counts for negative binomial distribution
    adata = datah_sample.X
    adata.layers["counts"] = adata.X.copy() # preserve counts
    # sc.pp.normalize_total(adata, target_sum=10e4)
    # sc.pp.log1p(adata)
    # adata.raw = adata # freeze the state in `.raw`
    scvi.data.setup_anndata(adata, layer="counts")

    ## Crate and train model
    model_args = {'use_cuda'     : True,
                  'n_latent'     : n_latent,
                  'n_layers'     : n_layers,
                  'dispersion'   : 'gene',
                  'n_hidden'     : n_hidden,
                  'n_hiddens'    : n_hidden_layers,
                  'dropout_rate' :  0,
                  'gene_likelihood'    :  'nb',
                  'log_variational'    :  True,
                  'latent_distribution':  'normal'
                  }

    if method == 'LDVAE':
        model = scvi.model.LinearSCVI(adata,**model_args)
    elif method == 'scVI':
        model = scvi.model.SCVI(adata,**model_args)
    else:
        raise Exception('Input valid scVI model')

    ## Train model
    model.train(n_epochs = n_epochs,
                lr       = lr,
                n_epochs_kl_warmup = n_epochs/2,
                metrics_to_monitor = ['reconstruction_error'],
                frequency = 1)

    ## Check train history
    df_history = {'reconstruction_error_test_set' : model.history['reconstruction_error_test_set'],
                  'reconstruction_error_train_set': model.history['reconstruction_error_train_set']}
    df_history = pd.DataFrame(df_history)
    df_history = pd.DataFrame(df_history.stack())
    df = df_history
    df.reset_index(inplace=True)
    df.columns = ['Epoch','Loss Type', 'Loss']
    df.to_csv(os.path.join(logdir_out,'history.csv'))
    sns.lineplot(data=df,x='Epoch',y='Loss', hue = 'Loss Type')
    plt.savefig(os.path.join(logdir_out,'loss.pdf'))
    plt.close()

    ## Extract the embedding space for scVI
    X_latent = model.get_latent_representation()
    y        = np.array(adata.obs['Cell Type'].tolist())

    # Save latent embeddings
    df    = pd.DataFrame(X_latent)
    df['Labels'] = y
    df.to_csv(os.path.join(logdir_out,'X_scvi.csv'))

    if method == 'LDVAE':
        loadings = model.get_loadings()
        loadings.to_csv(os.path.join(logdir_out,'X_loadings.csv'))

    ## Plot embedding space
    X_out,_ = reduce_dimensions(X_latent,
                                reduced_dimension = 2,
                                tsne_min=5,
                                method = 'UMAP')

    df = pd.DataFrame(X_out)
    df['Labels'] = y

    palette   = plot_args['palette']
    hue_order = plot_args['hue_order']
    df.columns = ['Dim 1', 'Dim 2', 'Label']
    ax = sns.scatterplot(data = df,
                    x = 'Dim 1',
                    y = 'Dim 2',
                    hue = 'Label',
                    s = 3,
                    edgecolor=None,
                    hue_order = hue_order,
                    palette = palette)
    ax.legend_.remove()
    plt.savefig(os.path.join(logdir_out,'embeddings.pdf'))
    plt.close()

    ## Perform classifier with 5-fold split
    skf = StratifiedKFold(n_splits=5)
    skf.get_n_splits(X_latent, y)
    results_clf_dict = {}
    results_clf = []
    for train_index, test_index in skf.split(X_latent, y):
        X_train = X_latent[train_index]
        X_test = X_latent[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        n_neighbors = len(np.unique(y)) * 2
        result = classifier.run_classifier(X_train = X_train, X_test = X_test,
                                  y_train = y_train, y_test = y_test,
                                  classifier = "KNeighborsClassifier", max_dim = 1e6,
                                  args = {'n_neighbors':n_neighbors})[0]
        results_clf.append(result)
    results_clf_dict['model'] = np.array(results_clf)
    df_clf_accuracy = pd.DataFrame(results_clf_dict)
    df_clf_accuracy.to_csv(os.path.join(logdir_out,'clf_accuracy.csv'),index=False)


# ==============================================================================
#                                 Run siVAE
# ==============================================================================

if method_run == 'siVAE':

    ## Iterate over k-fold
    for k_fold in range(int(max(k_split,1))):

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
                          'log_frequency': 100,
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
                          'l2_scale_final'     : 1e-3,
                          'log_variational'    : log_variational,
                          'beta_warmup'        : 1000,
                          'max_patience_count' : 100,
                          'zv_recon_scale'     : zv_recon_scale,
                          'use_batch'          : use_batch}

            graph_args['logdir_tf'] = logdir_out
            os.makedirs(logdir_out,exist_ok=True)
            ## Run model
            siVAE_result = run_VAE(graph_args_sample = graph_args,
                                   LE_method         = LE_method,
                                   datah_sample      = datah_sample,
                                   datah_feature     = datah_feature,
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
                X_latent = siVAE_result.get_sample_embeddings()
            elif model_type == 'sample':
                X_latent = siVAE_result['sample'].get_value('z_mu')

            X_out,_ = reduce_dimensions(X_latent,
                                        reduced_dimension = 2,
                                        tsne_min=5,
                                        method = 'UMAP')

            y = siVAE_result.get_value('labels')
            df = pd.DataFrame(X_out)
            df['Labels'] = y
            df.to_csv(os.path.join(logdir_out,'latent_{}.csv'.format(model_type)))
            df = pd.read_csv(os.path.join(logdir_out,
                                          'latent_{}.csv'.format(model_type)),index_col=0)

            ## Run 5-fold classifier experiment
            skf = StratifiedKFold(n_splits=5)
            skf.get_n_splits(X_latent, y)
            results_clf = []
            for train_index, test_index in skf.split(X_latent, y):
                X_train = X_latent[train_index]
                X_test = X_latent[test_index]
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
