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

# Model
from siVAE.model.run_model import run_VAE
from siVAE import analysis

from siVAE import FeatureImportance as FI

from load_MNIST import prepare_data

# ==============================================================================
#                           Specify Parameters for Model
# ==============================================================================

#### Specify method
LE_method = 'siVAE'

#### Specify directory
logdir = "out/MNIST/200716/test1"

#### Run Options
do_FA = False
save_result = False
overwrite_result = False

## Specify feature attribution Methods to be used
method_DE = ['SaliencyMaps','GradInput']

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
mb_size    = 0.1
l2_scale   = 5e-3
dataAPI    = False
keep_prob  = 1
lr         = 1e-3
early_stop = False
dr         = 0.8

#### Additional settings
# Architecture should be a string with a specific format
# architecture: "Encoder-LE-Decoder-Output (0)-Index of LE"
architecture = '1024-512-128-LE-128-512-1024-0-3'
decoder_activation = 'NA'
zv_recon_scale = 0.05
decoder_var    = 'scalar'
do_pretrain    = True
LE_dim = 20

test_name = 'zv_recon_scales'

var_list = [0.05,0.01,None]
var_name  = 'zv_recon_scale'

var_list2 = [0.1,0.2,0.5]
var_name2 = 'mb'

logdir_suffix=logdir

print("Run-{}-{}".format(var_name,var_list))
print("Run-{}-{}".format(var_name2,var_list2))

for var in var_list:

    zv_recon_scale = var
    logdir1 = os.path.join(logdir_suffix,test_name,'{}-{}'.format(var_name,var))

    for var2 in var_list2:

        mb_size = var2
        logdir = os.path.join(logdir1,'{}-{}'.format(var_name2,var2))

        result_dict_dir = os.path.join(logdir,'result_dict.pickle')

        #### Load MNIST Data
        kwargs_FI, sample_set, datah_feature, datah_sample, plot_args_dict_sf = prepare_data()

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

        # ==============================================================================
        #                                 Run Model
        # ==============================================================================

        graph_args['logdir_tf'] = logdir
        os.makedirs(logdir,exist_ok=True)

        ## Run model
        result_dict = run_VAE(logdir, graph_args, LE_method, datah_sample,
                              zv_recon_scale=zv_recon_scale, datah_feature=datah_feature,
                              do_pretrain=True, method_DE=method_DE,
                              sample_set=sample_set, do_FA = do_FA,
                              kwargs_FI = kwargs_FI)

        if save_result:
            util.save_pickle(result_dict, result_dict_dir)

        # ==============================================================================
        #                               Run Analysis
        # ==============================================================================

        analysis.save_losses(result_dict,logdir)

        ## Extract necessary values from the result of siVAE
        values_dict = analysis.extract_value(result_dict)

        palette   = plot_args_dict_sf['palette']
        hue_order = plot_args_dict_sf['hue_order']


        ## Plot scatter plots of sample-wise and feature-wise latent embeddings
        analysis.plot_latent_embeddings(values_dict, plot_args_dict_sf, palette,
                                        logdir=logdir, multidimension=True,
                                        show_legend=False, hue_order = hue_order,
                                        method_dim_reds = ['PCA'],
                                        s=5,edgecolor='none')


        # ==============================================================================
        #                           Run Analysis for Image
        # ==============================================================================

        ImageDims = plot_args_dict_sf['ImageDims']

        ## Visualization for Image Datasets (ImageDims is not None)
        ## Plot loadings
        analysis.plot_siVAE_loadings(values_dict, logdir, ImageDims)

        if 'Feature Attribution' in values_dict['model'].keys():
            analysis.plot_FA_loadings(values_dict, logdir, ImageDims)

        ## Plot recoded embeddings
        analysis.recode_embeddings(values_dict,
                                   plot_args_dict_sf,
                                   ImageDims, n_pc=3,
                                   logdir=logdir)

        ## Plot Feature Awareness
        analysis.plot_feature_awareness(result_dict,scaler,ImageDims,plot_args_dict_sf,logdir)
