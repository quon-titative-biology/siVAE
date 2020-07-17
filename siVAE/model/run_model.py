import os
import time
import logging

import tensorflow as tf

from . import VAE
from . import siVAE

from .analysis import run_FI
from siVAE import FeatureImportance as FI

import time
import numpy as np

def run_VAE(logdir, graph_args_sample, LE_method, datah_sample, datah_feature = None, do_pretrain = True,
            graph_args_feature = None, graph_args_siVAE = None, rs = 0, zv_recon_scale = 0,
            method_DE = ['SaliencyMaps','IntGrad','GradInput', 'Shapley', 'DeepLIFT'],
            sample_set = None, do_FA = False, kwargs_FI = {}):
    """ """

    method_DE_dict = {"SaliencyMaps": "Saliency Maps",
                      "GradInput"   : "Gradient * Input",
                      "IntGrad"     : "Integrated Gradients",
                      'Epsilon-LRP' : 'Epsilon-LRP',
                      'DeepLIFT'    : 'DeepLIFT',
                      'Occlusion'   : 'Occlusion',
                      'Shapley'     : 'Shapley Value'}

    method_DE = [ method_DE_dict[method] for method in method_DE ]

    if sample_set is not None:
        sample_input = sample_set['samples']
    else:
        sample_input = None

    ## Set up architecture of hidden layers
    LE_dim       = graph_args_sample['LE_dim']
    architecture = graph_args_sample['architecture']

    # For datah_sample
    h_dims = architecture.replace('LE',str(LE_dim))
    h_dims = [int(dim) for dim in h_dims.split("-")]
    datah_sample.index_latent_embedding = int(h_dims.pop())
    datah_sample.h_dims = h_dims

    # For datah_feature
    if datah_feature is not None:
        h_dims = architecture.replace('LE',str(LE_dim))
        h_dims = [int(dim) for dim in h_dims.split("-")]
        datah_feature.index_latent_embedding = int(h_dims.pop())
        _ = h_dims.pop(-2)
        datah_feature.h_dims = h_dims

    ## Run Model
    if LE_method == 'siVAE':

        # Graph argument for VAE2
        if graph_args_feature is None:
            graph_args_feature = dict(graph_args_sample)
            graph_args_feature['var_dependency'] = True
            graph_args_feature['X_mu_use_bias'] = True

        # Graph argument for combined model
        if graph_args_siVAE is None:
            graph_args_siVAE = dict(graph_args_sample)
            graph_args_siVAE['zv_recon_scale'] = zv_recon_scale

        result, result_s, result_f, model = siVAE.run(datah_sample, datah_feature,
                                                      graph_args_sample,
                                                      graph_args_feature,
                                                      graph_args_siVAE,
                                                      rs = rs,
                                                      do_pretrain    = do_pretrain,
                                                      sample_input   = sample_input,
                                                      do_FA         = do_FA,
                                                      method_DE      = method_DE,
                                                      kwargs_FI      = kwargs_FI)

        result_dict = {'sample':result_s,'feature':result_f,'model':result}

    else:

        if datah_sample.iterator is None:
            tf.reset_default_graph()

        config = graph_args_sample['config']

        with tf.Session(config = config) as sess:

            with FI.DeepExplain(session=sess) as de:

                model = VAE.AutoEncoder(data_handler = datah_sample, random_seed = rs, isVAE = True, name = 'VAE', **graph_args_sample)
                model.build_model(reset_graph = False)
                result = model.train(sess)

                ## Additional analysis
                if sample_input is not None:
                    sample_dict = {}
                    if len(method_DE) > 0 and do_FA:
                        logging.info('do_FA')
                        time.sleep(1)

                        feed_dict_full = {model.X  : sample_input,
                                          model.X_target : sample_input}

                        sample_z = model.calculate_z_mu(feed_dict_full)

                        feed_dict_base = {}
                        attributions_dict_all = {}

                        ## Encoder
                        attributions_dict = run_FI(input     = model.X,
                                                   target    = model.z_mu,
                                                   sample_DE = sample_input,
                                                   de        = de,
                                                   method_DE = method_DE,
                                                   feed_dict = feed_dict_base,
                                                   kwargs_FI = kwargs_FI)

                        attributions_dict_all['encoder'] = attributions_dict


                        ## Decoder
                        kwargs_FI2 = dict(kwargs_FI)
                        _ = kwargs_FI2.pop('baseline',None)
                        attributions_dict = run_FI(input     = model.z_sample,
                                                   target    = model.X_dist.mu,
                                                   sample_DE = sample_z,
                                                   de        = de,
                                                   method_DE = method_DE,
                                                   feed_dict = feed_dict_base,
                                                   kwargs_FI = kwargs_FI2)

                        attributions_dict_all['decoder'] = attributions_dict

                        sample_dict['attributions_samples'] = attributions_dict_all

                    result['sample_dict'] = sample_dict

        result_dict = {'model':result}

    result_dict['model']['metadata'] = np.array(datah_sample.X.obs_names)

    return result_dict
