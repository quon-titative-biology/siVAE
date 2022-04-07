## 2021/10/13
# Gene Regulatory Network Simulation Using Simple Multivariate Gaussian

## System
import os
import shutil

## Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import offsetbox
from matplotlib import cm
import seaborn as sns

import random
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

import tensorflow as tf

import network_class as ntwrk

from siVAE.data.data_handler import data2handler
from siVAE.run_model import run_VAE
from siVAE.util import reduce_dimensions
from siVAE.model.output import analysis
from siVAE.model.output.output_handler import get_feature_attributions

from exp211013.distributions import simulate_network
from exp211013.plots import plot_expressions, plot_scatter_states, plot_scatter_genes, plot_expressions_multiple

import itertools

import scanpy as sc

import logging
logging.getLogger('tensorflow').disabled = True
logging.getLogger().setLevel(logging.INFO)

## siVAE arguments
logdir = 'test41'

args_dict = {}
args_dict["LE_method"] = ['siVAE','VAE','LDVAE']
# args_dict["LE_method"] = ['siVAE']
# args_dict["LE_method"] = ['LDVAE']
# args_dict["zv_recon_scale"] = [0.05,0.01]
args_dict["zv_recon_scale"] = [0.05]
args_dict['l2_scale_final'] = [1e-4]
args_dict["beta"] = [1]
# args_dict['beta_feature'] = [0,0.1,0.5,1]
args_dict['beta_feature'] = [1]

## Larger architecture
args_dict["architecture"] = ['256-128-64-LE-64-128-256-0-3']
args_dict["LE_dim"] = [2,3,5,10,25,50]

## Smaller architecture
# args_dict["architecture"] = ['6-6-6-LE-6-6-6-0-3']
# args_dict["LE_dim"] = [3]

## Architecture
sample_size = 1000
n_tf = 3
group_size = 50
n_group = 5
rho = 0.8
percent_active = 0.2
network="Gaus"
logic='xor'
offset = 1
var1 = 0.1
var2 = 0.1

## Create logdirs
logdir = os.path.join(logdir,f'{network}-G{n_group}-g{group_size}-tf{n_tf}-logic{logic}-offset{offset}-var{var1}-{var2}')
os.makedirs(logdir, exist_ok=True)
logdir_out = os.path.join(logdir,'out')
os.makedirs(logdir_out,exist_ok=True)

########################### Custom inputs for siVAE ############################

LE_method = 'siVAE'

#### Set up tf config
gpu_device = '0'
os.environ["CUDA_VISIBLE_DEVICES"]  = gpu_device
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
config.allow_soft_placement = True
# config.intra_op_parallelism_threads = 5
# config.inter_op_parallelism_threads = 5

#### Model Parameters
iter       = 2000
mb_size    = 128
l2_scale   = 1e-3
dataAPI    = False
keep_prob  = 1
lr         = 1e-3
early_stop = False
dr         = 0.95
zv_recon_scale = 0

#### Additional settings
# Architecture should be a string with a specific format
# architecture: "Encoder-LE-Decoder-Output (0)-Index of LE"
architecture = '128-64-64-LE-64-64-128-0-3'
do_pretrain    = True
LE_dim = 2

output_distribution = 'normal'
log_variational =    False

graph_args = {'LE_dim'       : LE_dim,
              'architecture' : architecture,
              'config'       : config,
              'iter'         : iter,
              'mb_size'      : mb_size,
              'l2_scale'     : l2_scale,
              'dataAPI'      : dataAPI,
              'tensorboard'  : False,
              'batch_norm'   : False,
              'keep_prob'    : keep_prob,
              'log_frequency': 50,
              'learning_rate': lr,
              "early_stopping"   : early_stop,
              "validation_split" : 0,
              "decay_rate"       : dr,
              "decay_steps"      : 1000,
              'var_dependency'   : True,
              'activation_fun'   : None,
              'activation_fun_decoder': None,
              'output_distribution': output_distribution,
              'beta'               : 1,
              'l2_scale_final'     : 0,
              'log_variational'    : log_variational,
              'beta_warmup'        : 0.2,
              'max_patience_count' : 100,
              'zv_recon_scale'     : zv_recon_scale}

############################### simulate network ###############################

adata = simulate_network(sample_size=sample_size,
                         n_tf=n_tf,
                         group_size=group_size,
                         rho=rho,
                         percent_active=percent_active,
                         threshold=None,
                         network=network,
                         n_group = n_group,
                         logic = logic,
                         offset=offset,
                         var1=var1,
                         var2=var2)

adata.layers['raw'] = adata.X.copy()
# sc.pp.scale(adata)

## Define linkage based on scaled data
from scipy.cluster.hierarchy import linkage

X_in = adata.X[:200]
row_linkage = linkage(X_in)

## Initial visualization of dataset\
logdir_initial = os.path.join(logdir,'data')
for type_ in ['raw','scaled']:
    #
    if type_ == 'raw':
        X_in = adata.layers['raw']
    elif type_ == 'scaled':
        X_in = adata.X
    else:
        raise Exception('wrong type')
    ## Plot gene2gene correlation matrix
    corr = np.corrcoef(X_in.transpose())
    g = sns.heatmap(corr,center=0,cmap='RdBu')
    plt.savefig(os.path.join(logdir_data, f'corr-{type_}.pdf'))
    plt.close()
    ## Plot expressions as heatmap
    g = plot_expressions(adata=adata,
                         n_sample=len(row_linkage)+1,
                         row_cluster=True,
                         row_linkage=row_linkage)
    plt.savefig(os.path.join(logdir_data, f'exp-{type_}.pdf'))
    plt.close()
    plot_expressions_multiple(adata,
                              n_sample=len(row_linkage)+1,
                              row_cluster=True,
                              logdir=os.path.join(logdir_data, f'exp-{type_}.pdf'))
    ## Plot scatterplot annotated by state on/off
    X_emb_dict = {}
    for method in ['PCA','UMAP']:
        X_loadings,_ = reduce_dimensions(X_in,method=method)
        X_emb_dict[method] = X_loadings
    states = pd.DataFrame(adata.obs.State).copy().reset_index(drop=True)
    states.columns = [f"G{ii+1}" for ii in range(states.shape[1])]
    plot_scatter_states(X_emb_dict, states, os.path.join(logdir_data,f'scatter-{type_}.pdf'))

adata.obs['Labels'] = 'None'

################################# Run DR #######################################

X_feat_dict = {}
X_emb_dict = {}
X_recon_dict = {}
X_sample_dict = {}

#### VAE -----------------------------------------------------------------------

datah_sample, datah_feature, plot_args = data2handler(adata)
states = pd.DataFrame(datah_sample.X.obs.State).copy().reset_index(drop=True)
states.columns = [f"G{ii+1}" for ii in range(states.shape[1])]

logdir_VAE = os.path.join(logdir,"model")
os.makedirs(logdir_VAE,exist_ok=True)

for args in itertools.product(*args_dict.values()):
    args_update = {k:v for k,v in zip(args_dict.keys(),args)}
    logdir_tf = os.path.join(logdir_VAE,*[f"{k}-{v}" for ii,(k,v) in enumerate(args_update.items())])
    beta_feature = args_update.pop('beta_feature')
    graph_args.update(args_update)
    model_name = "-".join([f"{k}-{v}" for ii,(k,v) in enumerate(args_update.items())])
    # Remove and replace
    shutil.rmtree(logdir_tf, ignore_errors=True)
    os.makedirs(logdir_tf,exist_ok=True)
    graph_args['logdir_tf'] = logdir_tf
    #
    ## Run model
    if "LE_method" in graph_args.keys():
        LE_method = graph_args.pop("LE_method")
    ## Run model
    graph_args_feature = graph_args.copy()
    graph_args_feature['beta'] = beta_feature
    graph_args_feature['mb_size'] = 32
    graph_args_feature['lr'] = 1e-4
    graph_args_feature['grad_clipping'] = False
    #
    if LE_method == 'siVAE':
        additional_kwargs = {}
    else:
        sample_set = {'samples': datah_sample.X.X}
        additional_kwargs = {'do_FA'      : True,
                             'sample_set' : sample_set,
                             'method_DE'  : ['SaliencyMaps','GradInput','DeepLIFT']}
    siVAE_result = run_VAE(graph_args_sample = graph_args,
                           graph_args_feature= graph_args_feature,
                           LE_method         = LE_method,
                           datah_sample      = datah_sample,
                           datah_feature     = datah_feature,
                           do_pretrain       = True,
                           n_sample          = 1000,
                           **additional_kwargs)
    ##
    logdir_plot = os.path.join(logdir_tf,'plot')
    os.makedirs(logdir_plot,exist_ok=True)
    ##
    from siVAE.model.output.plot import plot_scalars
    analysis.save_losses(siVAE_result,logdir_tf)
    #
    # if graph_args['tensorboard']:
    #     plot_scalars(siVAE_result, logdir_plot)
    #
    # Extract feature embeddings
    if LE_method=="siVAE":
        X_loadings = siVAE_result.get_feature_embeddings()
        X_loadings,_ = reduce_dimensions(X_loadings,method='UMAP')
        X_feat_dict[model_name] = X_loadings
        X_feat_dict_temp = {'Feature emb': X_loadings}
    else:
        scores, method_DE = get_feature_attributions(siVAE_result)
        scores = scores['decoder']
        FA_loadings = analysis.infer_FA_loadings(np.swapaxes(scores,1,2),method=1)
        X_feat_dict_temp = {k:reduce_dimensions(v.transpose(),method='UMAP')[0] for k,v in zip(method_DE,FA_loadings)}
        X_feat_dict_temp['W'] = reduce_dimensions(siVAE_result.get_value('W').transpose(),method='UMAP')[0]
    plot_scatter_genes(X_feat_dict_temp,adata.var,logdir_plot,center=False)
    #
    # Extract sample embeddings
    if LE_method == "siVAE":
        X_emb = siVAE_result.get_sample_embeddings()
    else:
        X_emb = siVAE_result.get_value('z_mu')
    X_emb_red,_ = reduce_dimensions(X_emb, method="UMAP")
    X_emb_dict[model_name] = X_emb_red
    #
    # Extract reconstructions
    X_recon = siVAE_result.get_value('reconstruction')[1]
    X_recon_red,_ = reduce_dimensions(X_recon,method="UMAP")
    X_recon_dict[model_name]=X_recon_red
    #### Plots
    #
    ## Plot scatter
    plot_scatter_states({'Emb'  : X_emb_red,
                         'Recon': X_recon_red},
                        states, os.path.join(logdir_plot,f'scatter-emb-{"all"}.svg'))
    ## Plot recon expressions as heatmap
    g = plot_expressions(X=X_recon, obs=adata.obs, var=adata.var,
                         n_sample    = len(row_linkage)+1,
                         row_cluster = True,
                         row_linkage = row_linkage)
    plt.savefig(os.path.join(logdir_plot, f'exp-{"recon"}.pdf'))
    plt.close()
    plot_expressions_multiple(X=X_recon,obs=adata.obs,var=adata.var,
                              n_sample   = len(row_linkage)+1,
                              row_cluster= True,
                              logdir     = os.path.join(logdir_plot, f'exp-recon.pdf'))
    #
    ## Plot sampled expressions as heatmap
    X_sampled = siVAE_result.get_value('sample')
    X_sample_dict[model_name] = X_sampled
    obs = pd.DataFrame({"State":["Off"]*X_sampled.shape[0]})
    g = plot_expressions(X=X_sampled, obs=obs, var=adata.var,
                         n_sample=len(row_linkage)+1, row_cluster=True)
    plt.savefig(os.path.join(logdir_plot, f'exp-{"sampled"}.pdf'))
    plt.close()

#### Plot scatterplot of embeddings
X_in = datah_sample.X.X
for method in ['PCA','UMAP']:
    X_loadings,_ = reduce_dimensions(X_in,method=method)
    X_emb_dict[method] = X_loadings

plot_scatter_states(X_emb_dict, states, os.path.join(logdir,f'scatter-emb-{"all"}.svg'))
plot_scatter_states(X_recon_dict, states, os.path.join(logdir,f'scatter-recon-{"all"}.svg'))

#### Plot sampled spaces

#### PCA -----------------------------------------------------------------------

from sklearn.decomposition import PCA
#
for LE_dim in args_dict["LE_dim"]:
    logdir_plot = os.path.join(logdir,'pca',f'LE_dim-{LE_dim}')
    os.makedirs(logdir_plot,exist_ok=True)
    pca = PCA(LE_dim)
    X_train = datah_sample.dataset[0]
    X_test  = datah_sample.dataset[1]
    X_embedding = pca.fit_transform(X_train)
    X_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    X_feat_dict['PCA'] = X_loadings
    recon_errors={}
    for type_, X_in in zip(['train','test'],[X_train,X_test]):
        X_emb = pca.transform(X_in)
        X_recon = np.dot(X_emb, pca.components_) + pca.mean_
        recon_error = np.square(X_recon - X_in).sum(1).mean()
        recon_errors[type_]=[recon_error]
    #
    pd.DataFrame(recon_errors).to_csv(os.path.join(logdir_plot,'pca_recon_loss.csv'))
    ## Plot recon
    X_emb = pca.transform(datah_sample.X.X)
    X_recon = np.dot(X_emb, pca.components_) + pca.mean_
    #
    g = plot_expressions(X=X_recon, obs=adata.obs, var=adata.var, n_sample=200)
    plt.savefig(os.path.join(logdir_plot, f'exp-{"pca-recon"}.pdf'))
    plt.close()


########################## Plot feature embeddings #############################
## Transpose then DR

adata_t = adata.copy().transpose()
for m in ['PCA','tSNE','UMAP']:
    X_loadings,_ = reduce_dimensions(adata_t.X,method=m)
    X_feat_dict[f't-{m}'] = X_loadings

#### VAE
logdir_VAE = os.path.join(logdir,'VAE_t')
os.makedirs(logdir_VAE,exist_ok=True)
adata_t.obs['Labels'] = 'None'
datah_sample, datah_feature, plot_args = data2handler(adata_t)
graph_args['logdir_tf'] = logdir_VAE
## Run model
siVAE_result = run_VAE(graph_args_sample = graph_args,
                       LE_method         = 'VAE',
                       datah_sample      = datah_sample,
                       do_pretrain       = True,
                       sample_set        = None,
                       do_FA             = False,
                       kwargs_FI         = None,)
from siVAE.model.output.plot import plot_scalars
logdir_plot = os.path.join(logdir_VAE,'plot')
os.makedirs(logdir_plot,exist_ok=True)
# plot_scalars(siVAE_result, logdir_plot)
X_loadings = siVAE_result.get_model().get_dict()['z_mu']
X_loadings,_ = reduce_dimensions(X_loadings,method='UMAP')
X_feat_dict['t-VAE'] = X_loadings

plot_scatter_genes(X_feat_dict,adata_t.obs,logdir)
