import os
import time
import logging

import tensorflow as tf

from .model.VAE import VAE
from .model import siVAE

from .model.analysis import run_FI
from .model.output.output_handler import output_handler

from . import FeatureImportance as FI

import time
import numpy as np

def run_VAE(graph_args_sample, LE_method, datah_sample, datah_feature = None, do_pretrain = True,
            graph_args_feature = None, graph_args_siVAE = None, rs = 0,
            method_DE = ['SaliencyMaps','IntGrad','GradInput', 'Shapley', 'DeepLIFT'],
            sample_set = None, do_FA = False, kwargs_FI = {}, assign_dict={},
            num_input=None,num_output=None, n_sample=None):
    """
    run either VAE or siVAE and return the results in results_dict

    Parameters
    ----------
    graph_args_sample
        Arguments for sample-wise encoder/decoder
    graph_args_feature
        Arguments for feature-wise encoder/decoder
    graph_args_siVAE
        Arguments for combined model
    LE_method
        Model to use, either VAE or siVAE
    datah_sample
        data_handler object for sample-wise encoder/decoder
    datah_feature
        data_handler object for feature-wise encoder/decoder
    do_pretrain
        Set True to pre-train sample-wise and feature-wise encoder/decoder for siVAE
    method_DE
        List of feature attribution methods to use
        ['SaliencyMaps','IntGrad','GradInput', 'Shapley', 'DeepLIFT']
    sample_set
        Sample set to use for feature attribution
    do_FA
        Set True to run feature attribution
    kwargs_FI
        Keyword arguments for feature attribution
    """

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

    if LE_method == 'LDVAE':
        # Remove the decoder layers in architecture for LDVAE
        architecture = architecture.split('LE')
        architecture = architecture[0]+'LE-0'+architecture[1].split('-0')[1]
        graph_args_sample['architecture'] = architecture

    # For datah_sample
    h_dims = architecture.replace('LE',str(LE_dim))
    h_dims = [int(dim) for dim in h_dims.split("-")]
    datah_sample.index_latent_embedding = int(h_dims.pop())
    datah_sample.h_dims = h_dims

    ## Run Model
    config = graph_args_sample['config']

    # Set new graph
    if datah_sample.iterator is None:
        tf.reset_default_graph()

    graph = tf.get_default_graph()

    with tf.Session(config = config) as sess:

        if LE_method == 'siVAE':

            # Graph argument for feature-wise encoder/decoder
            if graph_args_feature is None:
                graph_args_feature = dict(graph_args_sample)
                graph_args_feature['var_dependency'] = True
                graph_args_feature['X_mu_use_bias'] = True
                graph_args_feature['output_distribution'] = 'normal'
                graph_args_feature['grad_clipping'] = True
                ##
                # graph_args_feature['beta'] = 0
                graph_args_feature['l2_scale_final'] = 0
                graph_args_feature['l2_scale'] = 1e-5
                graph_args_feature['use_batch'] = False
                ##
                # graph_args_feature['architecture'] = '2048-2048-2048-LE-2048-2048-0-3'
                # LE_dim = 2048
                # graph_args_feature['iter'] = 2000
                # graph_args_feature['learning_rate'] = 1e-3
                # graph_args_feature['decoder_var'] = 'deterministic'
                # graph_args_feature['isVAE'] = False

            # For datah_feature
            if datah_feature is not None:
                num_decoder_layers = len(architecture.split('LE-')[1].split("-"))-2
                h_dims = architecture.replace('LE',str(LE_dim))
                h_dims = [int(dim) for dim in h_dims.split("-")]
                datah_feature.index_latent_embedding = int(h_dims.pop())
                if num_decoder_layers > 0:
                    ## remove the last layer in the decoder to predict the W from sample
                    _ = h_dims.pop(-2)
                datah_feature.h_dims = h_dims

            # Graph argument for combined model
            if graph_args_siVAE is None:
                graph_args_siVAE = dict(graph_args_sample)
                # graph_args_siVAE['learning_rate'] = 5e-4

            result, result_s, result_f, model = siVAE.run(datah_sample, datah_feature,
                                                          graph_args_sample,
                                                          graph_args_feature,
                                                          graph_args_siVAE,
                                                          rs = rs,
                                                          do_pretrain  = do_pretrain,
                                                          sample_input = sample_input,
                                                          do_FA        = do_FA,
                                                          method_DE    = method_DE,
                                                          kwargs_FI    = kwargs_FI,
                                                          sess         = sess,
                                                          graph        = graph)

            result_dict = {'sample':result_s,'feature':result_f,'model':result}

        elif LE_method in ['VAE','LDVAE']:

            with FI.DeepExplain(session=sess) as de:

                model = VAE(data_handler = datah_sample,
                            random_seed  = rs,
                            name         = 'VAE',
                            **graph_args_sample)

                model.build_model(reset_graph = False)

                result = model.train(sess)
                result.get_dict()['var_names'] = datah_sample.X.var_names.to_numpy()

                for weights_name, weight in assign_dict.items():
                    if weights_name == 'W':
                        assign_op = self.W.assign(weight)
                        model.sess.run(assign_op)
                    if weights_name == 'b':
                        assign_op = self.b.assign(weight)
                        model.sess.run(assign_op)

                ## Additional analysis
                if sample_input is not None:
                    sample_dict = {}
                    if len(method_DE) > 0 and do_FA:
                        logging.info('Perform feature attribution')
                        print(sample_input.shape)
                        feed_dict_full = {model.X        : sample_input,
                                          model.X_target : sample_input}
                        # feed_dict_full = {model.X : sample_input}

                        sample_z = model.calculate_z_mu(feed_dict_full)

                        # feed_dict_base = {}
                        feed_dict_base = feed_dict_full
                        attributions_dict_all = {}

                        # ## Encoder
                        # attributions_dict = run_FI(input     = model.X,
                        #                            target    = model.z_mu,
                        #                            sample_DE = sample_input,
                        #                            de        = de,
                        #                            sess      = sess,
                        #                            method_DE = method_DE,
                        #                            feed_dict = feed_dict_base,
                        #                            kwargs_FI = kwargs_FI,
                        #                            mode      = 'reverse')
                        #
                        # attributions_dict_all['encoder'] = attributions_dict

                        # Whole model (Input to Output)
                        # attributions_dict = run_FI(input     = model.X,
                        #                            target    = model.X_dist.mu,
                        #                            sample_DE = sample_input,
                        #                            de        = de,
                        #                            sess      = sess,
                        #                            method_DE = method_DE,
                        #                            feed_dict = feed_dict_base,
                        #                            kwargs_FI = kwargs_FI,
                        #                            mode      = 'forward')
                        #
                        # attributions_dict_all['whole'] = attributions_dict

                        ## Decoder
                        kwargs_FI2 = dict(kwargs_FI)
                        _ = kwargs_FI2.pop('baseline',None)
                        attributions_dict = run_FI(input     = model.z_sample,
                                                   target    = model.X_dist.mu,
                                                   sample_DE = sample_z,
                                                   de        = de,
                                                   sess      = sess,
                                                   method_DE = method_DE,
                                                   feed_dict = feed_dict_base,
                                                   kwargs_FI = kwargs_FI2,
                                                   mode      = 'reverse')

                        attributions_dict_all['decoder'] = attributions_dict

                        #### Duration
                        # logging.info('Perform feature attribution: Test')
                        # kwargs_FI2 = dict(kwargs_FI)
                        # mode = 'forward'
                        # # mode = 'reverse'
                        # _ = kwargs_FI2.pop('baseline', None)
                        # feed_dict_base = feed_dict_full
                        # input = model.z_sample
                        # target = model.X_dist.mu
                        # sample_DE = sample_z
                        #
                        # if mode == 'forward':
                        #     n_node = int(input.shape[1])
                        # else:
                        #     n_node = int(target.shape[1])
                        #
                        # durations = []
                        # start = time.time()
                        # for ii in range(10):
                        #     logging.info('Node-{}/{}'.format(ii+1,n_node))
                        #     masks = [np.arange(n_node) == ii]
                        #     print(len(masks))
                        #
                        #     attributions_dict = run_FI(input     = input,
                        #                                target    = target,
                        #                                sample_DE = sample_DE,
                        #                                de        = de,
                        #                                sess      = sess,
                        #                                method_DE = method_DE,
                        #                                feed_dict = feed_dict_base,
                        #                                kwargs_FI = kwargs_FI2,
                        #                                mode      = mode,
                        #                                masks     = masks)
                        #
                        #     durations.append(time.time()- start)
                        #
                        #     print(time.time()- start)
                        #
                        # attributions_dict_all['durations'] = np.array(durations)
                        # attributions_dict_all['mode'] = mode
                        # attributions_dict_all['decoder'] = attributions_dict

                        sample_dict['attributions_samples'] = attributions_dict_all

                    result.get_dict()['sample_dict'] = sample_dict

            result_dict = {'model':result}

        ## Convert to output_handler
        result_dict = output_handler(result_dict)

        if n_sample is not None:
            sampled = model.sample(n_sample)
            result_dict.get_model().add_result('sample', sampled)

    result_dict.get_model().add_result('metadata' ,np.array(datah_sample.X.obs_names))
    result_dict.get_model().add_result('obs_names',np.array(datah_sample.X.obs_names))
    result_dict.get_model().add_result('var_names',np.array(datah_sample.X.var_names))
    result_dict.get_model().add_result('obs',np.array(datah_sample.X.obs))
    result_dict.get_model().add_result('var',np.array(datah_sample.X.var))

    return result_dict
