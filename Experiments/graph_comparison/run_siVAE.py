## System
import os,gc
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
from siVAE.data import data_handler as dh

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

parser = argparse.ArgumentParser()

parser.add_argument('--logdir' , type=str, default='out')
parser.add_argument('--datadir',type=str, default='default')
parser.add_argument('--index_dir',type=str, default='none')

parser.add_argument('--method' , type=str, default='siVAE')
parser.add_argument('--do_FA'  , type=str2bool, default=False)
parser.add_argument('--reduce_mode',type=str, default='sample')
parser.add_argument('--num_reduced',type=int, default=10000)
parser.add_argument('--reduce_subset',type=str, default="All")
parser.add_argument('--datadirbase',type=str, default='default')
parser.add_argument('--use_full_data',type=bool, default=True)
parser.add_argument('--use_batch',type=bool, default=False)

parser.add_argument('--LE_dim',type=int, default=2)
parser.add_argument('--k_split',type=float, default=0.8)

parser.add_argument('--mb_size',type=float, default=128)
parser.add_argument('--iter',type=int, default=2000)
parser.add_argument('--lr',type=float,default=1e-3)

parser.add_argument('--zv_recon_scale', type=float, default=0.05)
parser.add_argument('--l2_scale', type=float, default=1e-3)

parser.add_argument('--activation_fun', type=str, default='relu')

args = parser.parse_args()

# Directories
logdir   = args.logdir
datadir  = args.datadir
datadirbase = args.datadirbase
index_dir = args.index_dir

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
lr      = args.lr

# Model parameters
zv_recon_scale = args.zv_recon_scale
l2_scale       = args.l2_scale
use_batch      = args.use_batch

# Model setup
activation_fun = args.activation_fun
str2act = {'relu'      : tf.nn.relu,
           'leaky_relu': tf.nn.leaky_relu,
           'linear'    : None,
           'elu'       : tf.nn.elu,
           'sigmoid'   : tf.math.sigmoid}
activation_fun = str2act[activation_fun]

# ==============================================================================

os.makedirs(logdir,exist_ok=True)

#### Run Options
save_result = True
overwrite_result = False

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
dataAPI    = False
keep_prob  = 1
early_stop = False
dr         = 0.9

#### Additional settings
# Architecture should be a string with a specific format
# architecture: "Encoder-LE-Decoder-Output (0)-Index of LE"
architecture = '1024-512-128-LE-128-512-1024-0-3'
decoder_activation = 'NA'
decoder_var    = 'scalar'
do_pretrain    = True

# Method changing
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

# ==============================================================================
#                              Prepare data_handlers for siVAE
# ==============================================================================

## Load data

if '.h5ad' in datadir:

    if index_dir is not 'none':
        # e.g.
        # convert 'path/experiment/task-0/scaled.h5ad' to
        # 'path/experiment/task/scaled.h5ad''
        prefix,filename = datadir.rsplit('/',1)
        prefix,kfold = prefix.rsplit('-',1)
        datadir = os.path.join(prefix,filename)

    adata = sc.read_h5ad(datadir)

    if index_dir is not 'none':
        # e.g.
        # convert 'path/experiment/task-0/ndex.npy' to
        # 'path/experiment/task/index.npy'
        prefix,filename = index_dir.rsplit('/',1)
        prefix,kfold = prefix.rsplit('-',1)
        index_dir = os.path.join(prefix,filename)
        idx = np.load(index_dir)[int(kfold)]
        adata = adata[idx]

elif '.npy' in datadir:
    data = np.load(datadir)
    adata = sc.AnnData(X=data)

if 'celltype' in adata.obs.columns:
    adata.obs['Labels'] = adata.obs['celltype']
else:
    adata.obs['Labels'] = 'None'

adata.var['Labels'] = 'None'

#### Create data handler object for both sample and feature
datah_sample  = dh.adata2datah(adata, mode = 'sample')
datah_sample.create_split_index_list(k_split=k_split,random_seed=0)
datah_feature = dh.adata2datah(adata, mode = 'feature',
                               num_reduced   = num_reduced,
                               reduce_mode   = reduce_mode)

if use_batch:
    assert 'batch' in adata.obsm.keys(), 'If use_batch=True, provide batch array'
    datah_sample.batch = adata.obsm['batch']

# ==============================================================================
#                                 Run siVAE
# ==============================================================================


if method_run == 'siVAE':

    ## Iterate over k-fold
    for k_fold in range(int(max(k_split,1))):

        logdir_out = os.path.join(logdir,'kfold-{}'.format(k_fold))
        os.makedirs(logdir_out,exist_ok=True)
        siVAE_result_dir = os.path.join(logdir_out,'siVAE_result.pickle')

        datah_sample.create_dataset(kfold_idx=k_fold)

        if not overwrite_result and os.path.isfile(siVAE_result_dir):

            siVAE_result = util.load_pickle(siVAE_result_dir)

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
                          'l2_scale_final'     : 1e-4,
                          'log_variational'    : log_variational,
                          'beta_warmup'        : 1000,
                          'max_patience_count' : 100,
                          'zv_recon_scale'     : zv_recon_scale,
                          'use_batch'          : use_batch}

            graph_args['logdir_tf'] = logdir_out

            ## Run model
            siVAE_result = run_VAE(graph_args_sample = graph_args,
                                   LE_method         = LE_method,
                                   datah_sample      = datah_sample,
                                   datah_feature     = datah_feature,
                                   do_pretrain       = True,
                                   do_FA             = False)

            analysis.save_losses(siVAE_result,logdir_out)

            if save_result:
                util.save_pickle(siVAE_result, siVAE_result_dir)

            from siVAE.model.output.plot import plot_scalars
            logdir_plot = os.path.join(logdir,'plot')
            os.makedirs(logdir_plot,exist_ok=True)
            plot_scalars(siVAE_result, logdir_plot)
