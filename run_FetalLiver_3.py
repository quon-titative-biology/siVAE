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

from load_FetalLiver import prepare_data


# ==============================================================================
#                           Specify Parameters for Model
# ==============================================================================

#### Specify method
LE_method = 'siVAE'

#### Specify directory
logdir = "out/FetalLiver/subset_all/200707/test5"
# logdir = "out/HumanLiver/subset_1/200703/test1"

#### Run Options
do_FA = False
save_result = False
overwrite_result = False

## Specify feature attribution Methods to be used
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

        if not overwrite_result and os.path.isfile(result_dict_dir):

            result_dict = util.load_pickle(result_dict_dir)
            plot_args_dict_sf['sample']['labels'] = result_dict['model']['labels'][0]

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

            genesets = {"GO:BP":"/home/yongin/projects/siVAE/data/MSigDB/c5.bp.v7.1.symbols.gmt",
                        "scsig":"/home/yongin/projects/siVAE/data/MSigDB/scsig.all.v1.0.1.symbols.gmt",
                        "Hallmark":"/home/yongin/projects/siVAE/data/MSigDB/h.all.v7.1.symbols.gmt",
                        "KEGG":"/home/yongin/projects/siVAE/data/MSigDB/c2.cp.kegg.v7.1.symbols.gmt"}

            genesets = None

            palette   = plot_args_dict_sf['palette']
            hue_order = plot_args_dict_sf['hue_order']
            ImageDims = plot_args_dict_sf['ImageDims']

            ## Plot scatter plots of sample-wise and feature-wise latent embeddings
            analysis.plot_latent_embeddings(values_dict, plot_args_dict_sf, palette,
                                            logdir=logdir, multidimension=True,
                                            show_legend=False, hue_order = hue_order,
                                            method_dim_reds = ['PCA'],
                                            s=5,edgecolor='none')

            ## Plot contribution of genes to latent dimensions based on loadings
            analysis.loading_analysis(values_dict, plot_args_dict_sf, logdir, genesets=genesets,
                                      dims = LE_dim, num_genes = 20, color = 'red')

            ## Perform PVE analysis
            analysis.PVE_analysis(values_dict,save_result=True,logdir=logdir)

            ## Plot feature awareness of genes based on siVAE/VAE
            FeatureAwareness = analysis.get_feature_awareness(values_dict,
                                                              plot_args_dict_sf=plot_args_dict_sf,genesets=genesets,
                                                              logdir=logdir, num_genes=20, color = 'red')

            ## Plot feature awareness of genes based on PCA
            FeatureAwareness_pca = analysis.get_feature_awareness_pca(values_dict,
                                                              plot_args_dict_sf=plot_args_dict_sf,
                                                              logdir=logdir, num_genes=20, color = 'red')

            ## Compare loadings
            analysis.compare_loadings(values_dict,logdir)

            ## Custom --------------------------------------------------------------

            metadata = plot_args_dict_sf['sample']['metadata'].transpose()
            labels = metadata[0]
            weeks = metadata[2]
            X = values_dict['model']['z_mu'][:,0]
            Y = values_dict['model']['z_mu'][:,1]
            weeks_in_list = [['7 weeks gestation','8 weeks gestation'],
                             ['9 weeks gestation','11 weeks gestation'],
                             ['12 weeks gestation','13 weeks gestation','14 weeks gestation'],
                             ['16 weeks gestation','17 weeks gestation']]

            for ii,weeks_in in enumerate(weeks_in_list):
                df_plot = pd.DataFrame({'X':X,'Y':Y,'Weeks':weeks,'Label':labels})
                df_plot = df_plot[np.isin(df_plot['Weeks'],weeks_in)]
                kwargs={'s':5,'edgecolor':'black'}
                # kwargs={'s':20}
                # kwargs={}
                ax = sns.scatterplot(x = "X", y = "Y", hue = "Label", data = df_plot, palette = palette, hue_order = hue_order,linewidth=0.05, **kwargs)
                ax.legend_.remove()
                ax.set_xlim(-3.5,2)
                ax.set_ylim(None,2)
                figname = os.path.join(logdir,"ScatterLE-byweeks_{}.pdf".format(ii+1))
                plt.savefig(figname)
                plt.close()

            # 2
            metadata = plot_args_dict_sf['sample']['metadata'].transpose()
            metadata_names = plot_args_dict_sf['sample']['metadata_name']
            kwargs={'s':5,'edgecolor':"none"}
            show_legend = False
            data_type = 'Original'
            X = values_dict['model']['X']
            X_recon = values_dict['model']['X_recon']
            X_dict = {'Reconstruction':X_recon, 'Original':X}
            for data_type,X in X_dict.items():
                print(data_type)
                for method_dim_red in ['PCA']:
                    print(method_dim_red)
                    dim = np.min([2,LE_dim])
                    X_plot, dim_labels = reduce_dimensions(X, reduced_dimension = dim, method = method_dim_red)
                    for metadata_name, labels_test in zip(metadata_names,metadata):
                        df_plot = pd.concat([pd.DataFrame(X_plot), pd.DataFrame(labels_test)], axis = 1)
                        df_plot.columns = dim_labels + ["Label"]
                        if X_plot.shape[-1] == 1:
                            df_plot['pc_1'] = 0
                            dim_labels.append('pc_1')
                        # Plot
                        if len(np.unique(labels_test)) < 30:
                            df_plot = df_plot.sort_values(by="Label")
                            if metadata_name == 'Cell Type':
                                ax = sns.scatterplot(x = dim_labels[0], y = dim_labels[1], hue = "Label", data = df_plot, palette = palette, hue_order = hue_order, **kwargs)
                            else:
                                ax = sns.scatterplot(x = dim_labels[0], y = dim_labels[1], hue = "Label", data = df_plot, **kwargs)
                            figname = os.path.join(logdir,"ScatterLE-{}_{}_{}.pdf".format(data_type,method_dim_red,metadata_name))
                            plt.savefig(figname)
                            plt.close()

            # 3
            kwargs={'s':5,'edgecolor':"none"}
            X = adata.X
            dim=2
            method_dim_red = 'tSNE'
            X_plot, dim_labels = reduce_dimensions(X, reduced_dimension = dim, method = method_dim_red)
            labels_test = adata.obs['Labels'].values
            df_plot = pd.concat([pd.DataFrame(X_plot), pd.DataFrame(labels_test)], axis = 1)
            df_plot.columns = dim_labels + ["Label"]
            ax = sns.scatterplot(x = dim_labels[0], y = dim_labels[1], hue = "Label", data = df_plot, palette = palette, hue_order = hue_order, **kwargs)
            # ax.set_ylim(-10,30)
            ax.legend_.remove()
            plt.savefig(os.path.join(logdir,'tsne.pdf'))
            plt.close()

            # ==============================================================================
            #                           Run Analysis for Image
            # ==============================================================================

## Visualization for Image Datasets (ImageDims is not None)
# ## Plot loadings
# analysis.plot_siVAE_loadings(values_dict, logdir, ImageDims)
#
# if 'Feature Attribution' in values_dict['model'].keys():
#     analysis.plot_FA_loadings(values_dict, logdir, ImageDims)
#
# ## Plot recoded embeddings
# analysis.recode_embeddings(values_dict,
#                            plot_args_dict_sf,
#                            ImageDims, n_pc=3,
#                            logdir=logdir)
#
# ## Plot Feature Awareness
# analysis.plot_feature_awareness(result_dict,scaler,ImageDims,plot_args_dict_sf,logdir)
