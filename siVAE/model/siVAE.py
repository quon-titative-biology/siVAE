import os
import time
import logging

# Tensorflow
import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
import pandas as pd
import math

from .layer import layer
from .layer import KL_divergence_normal
from .layer import create_variance
from .layer import Distribution
from .util import batch_handler
from . import VAE
from .analysis import run_FI

from siVAE import FeatureImportance as FI
from siVAE import util

import scanpy as sc


def create_placeholder_W(VAE_sample):
    """
    Input: self.VAE_sample
    Return: placeholder (matrix of zeros) with
    """

    preW = VAE_sample.W
    preW_shape = [int(s) for s in preW.shape]
    preW_placeholder = np.zeros(preW.shape).transpose()

    return preW_placeholder


def setup_siVAE(datah_in, datah_in_transpose, graph_args1, graph_args2, graph_args3,
                graph = None, rs = 1):
    """ Set up siVAE """

    times = {}
    duration = {}
    times['start'] = time.time()

    if graph is None:
        graph = tf.get_default_graph()

    VAE_sample = VAE.AutoEncoder(data_handler = datah_in,
                                 random_seed  = rs,
                                 isVAE        = True,
                                 name         = 'VAE_sample',
                                 **graph_args1)
    VAE_sample.build_model(reset_graph = False, graph = graph)
    times['VAE_sample_build'] = time.time()
    ## Set datah_in_transpose for VAE_feature
    preW = VAE_sample.W
    preW_shape = [int(s) for s in preW.shape]
    preW_placeholder = np.zeros(preW.shape).transpose()

    datah_in_transpose.y = sc.AnnData(preW_placeholder)
    k_split = 1
    datah_in_transpose.create_split_index_list(k_split, random_seed = rs)
    ks = 0
    datah_in_transpose.create_dataset(kfold_idx = ks)

    ## Create VAE_feature
    VAE_feature = VAE.AutoEncoder(data_handler = datah_in_transpose,
                           random_seed = rs, isVAE = True,
                           name = 'VAE_feature', **graph_args2)

    VAE_feature.build_model(reset_graph = False, graph = graph)
    times['VAE_feature_build'] = time.time()

    VAE_dict = {'sample' : VAE_sample,
                'feature': VAE_feature}

    ## Create siVAE
    model = combinedVAE(random_seed = rs, name = 'combinedModel', **graph_args3)
    model.combine_models(VAE_dict)
    times['VAE_feature_build'] = time.time()

    return model, VAE_sample, VAE_feature


def run(datah_in, datah_in_transpose, graph_args1, graph_args2, graph_args3,
        rs = 1, do_pretrain = True,
        sample_input = None, do_FA = False, method_DE = [], kwargs_FI = {}):

    """ run """

    start_time = time.time()
    times = {}
    times['start'] = time.time()

    if datah_in.iterator is None:
        tf.reset_default_graph()
    graph = tf.get_default_graph()

    config = graph_args1['config']

    with tf.Session(config = config) as sess:

        with FI.DeepExplain(session=sess) as de:

            model, VAE_sample, VAE_feature = setup_siVAE(datah_in, datah_in_transpose,
                                            graph_args1, graph_args2, graph_args3,
                                            rs = rs, graph = graph)

            times['build_model'] = time.time()

            ## Train
            results1 = None
            results2 = None

            logdir_tf = graph_args1['logdir_tf']

            ## Initialize
            tf.set_random_seed(VAE_sample.random_seed)
            tf.logging.set_verbosity(tf.logging.FATAL)
            sess.run(tf.global_variables_initializer())

            var_init = sess.run([VAE_sample.X_dist.var, VAE_sample.z_dist.var],
                                feed_dict = {VAE_sample.X: VAE_sample.data_handler.X.X})

            ## Initialize tensorboard
            logdir_train = os.path.join(logdir_tf,'train')
            logdir_test  = os.path.join(logdir_tf,'test')
            train_writer = tf.summary.FileWriter(logdir_train, sess.graph)
            test_writer  = tf.summary.FileWriter(logdir_test, sess.graph)

            if do_pretrain:
                ## Train VAE_sample
                logging.info("Pre-train sample-wise encoder and decoder")
                results1 = VAE_sample.train(sess = sess,
                                            initialize = False,
                                            train_writer = train_writer,
                                            test_writer = test_writer,
                                            close_writer = False)

                times['train_VAE_sample'] = time.time()

                ## Train VAE_feature
                logging.info("Pre-train feature-wise encoder and decoder")
                preW = results1['W'].transpose()
                VAE_feature.data_handler.y = sc.AnnData(preW)
                ks = 0
                VAE_feature.data_handler.create_dataset(kfold_idx = ks)

                results2 = VAE_feature.train(sess = sess,
                                             initialize = False,
                                             train_writer = train_writer,
                                             test_writer = test_writer,
                                             close_writer = False)
                times['train_VAE_feature'] = time.time()

            else:
                VAE_sample.sess  = sess
                VAE_feature.sess = sess
                times['train_VAE_sample']  = time.time()
                times['train_VAE_feature'] = time.time()

            ## Train model
            results = model.train(sess = sess, initialize = False, close_writer = True,
                                  train_writer = train_writer, test_writer = test_writer)

            times['train_combined'] = time.time()

            results['duration'] = time.time() - start_time
            results['time']     = times

            ## Additional Analysis
            if sample_input is not None:

                ## Sample
                feed_dict_full = {model.VAE_sample.X        : sample_input,
                                  model.VAE_sample.X_target : sample_input,
                                  model.VAE_feature.X       : model.VAE_dict['feature'].data_handler.X.X}

                z_mu          = model.VAE_sample.calculate_z_mu(feed_dict_full)
                sample_output = np.array(model.reconstruct_X(feed_dict_full))

                sample_dict = {"input" : sample_input,
                               "output": sample_output}

                if model.zv_recon_scale is not None:
                    sample_output_hl = model.reconstruct_X_hl(feed_dict_full)

                sampled_outputs = np.array([model.reconstruct_X(feed_dict_full)])
                sampled_outputs = np.swapaxes(sampled_outputs,0,1) # swap 0:Sample 1: Image

                ## Decoder layers
                decoder_layers_list = []
                for ii in range(len(sampled_outputs)):

                    sampled_output = sampled_outputs[ii,]

                    feed_dict_full = {model.VAE_sample.X        : sampled_output,
                                      model.VAE_sample.X_target : sampled_output,
                                      model.VAE_feature.X       : model.VAE_dict['feature'].data_handler.X.X}

                    z_mu = model.VAE_sample.calculate_z_mu(feed_dict_full)
                    X_mu = np.array(model.reconstruct_X(feed_dict_full)) # Don't sample X
                    decoder_layers     = model.VAE_sample.calculate_decoder_layers(feed_dict_full)
                    decoder_layers     = [z_mu] + decoder_layers
                    decoder_layers[-1] = X_mu
                    decoder_layers_list.append(decoder_layers)

                sample_dict['decoder_layers'] = decoder_layers_list

                if model.zv_recon_scale is not None:
                    sample_dict['output_hl'] = sample_output_hl

                if do_FA and len(method_DE) > 0:

                    logging.info('Perform feature attribution')

                    feed_dict_full = {model.VAE_sample.X        : sample_input,
                                      model.VAE_sample.X_target : sample_input,
                                      model.VAE_feature.X       : model.VAE_dict['feature'].data_handler.X.X}

                    sample_z = model.VAE_sample.calculate_z_mu(feed_dict_full)

                    feed_dict_base = {model.VAE_sample.X_target : sample_input,
                                      model.VAE_feature.X       : model.VAE_dict['feature'].data_handler.X.X}

                    attributions_dict_all = {}

                    ## Encoder
                    attributions_dict = run_FI(input     = model.VAE_sample.X,
                                               target    = model.VAE_sample.z_mu,
                                               sample_DE = sample_input,
                                               de        = de,
                                               method_DE = method_DE,
                                               feed_dict = feed_dict_base,
                                               kwargs_FI = kwargs_FI)

                    attributions_dict_all['encoder'] = attributions_dict

                    ## Decoder
                    kwargs_FI2 = dict(kwargs_FI)
                    _ = kwargs_FI2.pop('baseline',None)
                    attributions_dict = run_FI(input     = model.VAE_sample.z_sample,
                                               target    = model.X_dist.mu,
                                               sample_DE = sample_z,
                                               de        = de,
                                               method_DE = method_DE,
                                               feed_dict = feed_dict_base,
                                               kwargs_FI = kwargs_FI2)

                    attributions_dict_all['decoder'] = attributions_dict

                    sample_dict['attributions_samples'] = attributions_dict_all

                results['sample_dict'] = sample_dict

    return results, results1, results2, model


class combinedVAE(object):

    def __init__(self, logdir_tf = ".", data_handler = None, iter = 5000,
                 h_dims = None, LE_dim = None, architecture = None,
                 mb_size = 1000, learning_rate = 1e-3, l2_scale = 0.0, l1_scale = 0.0,
                 early_stopping = 0, tolerance = 0, min_early_stopping = 0,
                 activation_fun = tf.nn.relu, random_seed = 0,
                 log_frequency = 100, batch_norm = False, keep_prob = None, masking = False,
                 dataAPI = False, tensorboard = False, metadata = False, permute_axis = -2,
                 custom = False, config = None, validation_split = 0,
                 decay_rate=1, decay_steps = 1000, save_recon = True, save_LE = True,
                 decoder_var='scalar', save_W=True, set_y_logvar=False, decoder_activation = None,
                 name="", var_dependency=True, optimizer_type=tf.compat.v1.train.AdamOptimizer,
                 kernel_initializer=tf.contrib.layers.xavier_initializer(uniform = False), X_mu_use_bias=True,
                 zv_recon_scale=None, hl_recon_scale=None):

        self.decoder_var = decoder_var
        self.custom = custom
        self.var_dependency = var_dependency
        self.X_mu_use_bias = X_mu_use_bias
        self.zv_recon_scale = zv_recon_scale
        self.hl_recon_scale = hl_recon_scale

        self.decoder_activation = decoder_activation

        ## Recalculate h_dims if h_dims = 0
        self.save_LE = save_LE
        self.save_W = save_W
        self.set_y_logvar = set_y_logvar
        self.save_recon = save_recon

        ## Variable scope name
        self.name = name

        ## Set activation function
        self.activation_fun = activation_fun

        ## Training Parameters
        self.mb_size = mb_size
        self.iter = iter
        self.keep_prob = keep_prob

        ## Hyperparameters
        self.random_seed = random_seed

        # learning rate
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

        # Regularization
        self.l1_scale = l1_scale
        self.l2_scale = l2_scale
        self.batch_norm = batch_norm
        self.placeholder_reg = {}

        ## Early stopping
        self.validation_split = validation_split
        self.tolerance = tolerance
        self.early_stopping = early_stopping
        self.min_early_stopping = min_early_stopping
        self.iter_min_early_stopping = self.iter * self.min_early_stopping

        ## Tensorboard
        self.tensorboard = tensorboard
        self.logdir_tf = logdir_tf
        self.metadata = metadata
        self.summary_list = []

        self.feed_dict = {}

        if self.metadata and not self.tensorboard:
            logging.info("Tensorboard has been switched to true as metadata requires tensorboard")
            self.tensorboard = True

        self.log_frequency = log_frequency
        self.permute_axis = permute_axis
        self.dataAPI = dataAPI
        self.config = config


    def combine_models(self, VAE_dict):
        """
        """
        self.VAE_dict = VAE_dict
        assert all(VAE_type in list(VAE_dict.keys())  for VAE_type in ['sample', 'feature']), 'VAE_dict must contain both cell and gene type'

        self.VAE_sample  = self.VAE_dict['sample']
        self.VAE_feature = self.VAE_dict['feature']

        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):

            with tf.name_scope('W'):
                self.W = tf.transpose(self.VAE_feature.X_mu)

            with tf.name_scope('y'):
                self.y = tf.identity(self.VAE_sample.y_sample)

            if self.X_mu_use_bias:
                with tf.name_scope('b'):
                    self.b = tf.identity(self.VAE_sample.b)

            with tf.name_scope('std'):
                logging.info('self.VAE_sample.X_dist.std:')
                logging.info(self.VAE_sample.X_dist.std)
                self.X_std = tf.identity(self.VAE_sample.X_dist.std)

        logging.info('VAE_sample')
        logging.info(self.VAE_sample)
        logging.info(self.W)

        with tf.variable_scope(self.VAE_sample.output_variable_scope, reuse = True):

            prob_layer,_,_ = self.VAE_sample.build_output_layer(self.y,
                                                                fun = self.VAE_sample.activation_fun,
                                                                var_type = self.VAE_sample.decoder_var,
                                                                var_dependency = self.VAE_sample.var_dependency,
                                                                l1_scale = self.VAE_sample.l1_scale,
                                                                l2_scale = self.VAE_sample.l2_scale,
                                                                use_bias = self.VAE_sample.X_mu_use_bias,
                                                                W = self.W, b = self.b, h_std = self.X_std,
                                                                custom = True)
            self.X_dist = prob_layer

            ## For latent embedding layer
            with tf.name_scope('X_mu_zv'):
                self.z = self.VAE_sample.z_sample
                self.v = self.VAE_feature.z_sample

                if self.zv_recon_scale is not None:
                    self.X_mu_zv = tf.matmul(self.z,tf.transpose(self.v))
                    if self.X_mu_use_bias:
                        self.X_mu_zv = tf.add(self.X_mu_zv, self.b)

            ## For hidden layers
            if self.hl_recon_scale is not None:
                self.X_mu_hl = []
                sample_layers = self.VAE_sample.decoder_layers[:-1]
                feature_layers = self.VAE_feature.decoder_layers[:-1]
                for ii,(l_sample,l_feature) in enumerate(zip(sample_layers,feature_layers)):
                    with tf.name_scope('X_mu_' + str(ii)):
                        X_mu_new = tf.matmul(l_sample,tf.transpose(l_feature))
                        if self.X_mu_use_bias:
                            X_mu_new = tf.add(X_mu_new, self.b)
                    self.X_mu_hl.append(X_mu_new)

        self.summary_list += self.VAE_sample.summary_list
        # self.summary_list += self.VAE_feature.summary_list # merging the feature summary list requires input to X_target placeholder


        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            self.create_loss()


    def create_loss(self):
        """
        Crate losses
        """

        ## Add all losses
        self.decoder_loss = -self.X_dist.dist.log_prob(self.VAE_sample.X_target)
        self.decoder_loss_scalar = tf.reduce_mean(self.decoder_loss)
        self.recon_loss_per_sample = tf.square(self.VAE_sample.X_target - self.X_dist.mu) * int(self.VAE_sample.X_target.shape[-1])
        self.recon_loss = tf.losses.mean_squared_error(self.VAE_sample.X_target, self.X_dist.mu) * int(self.VAE_sample.X_target.shape[-1])

        ## Total loss consists of KL
        self.total_loss = self.VAE_sample.KL_loss_scalar + \
                          self.VAE_feature.KL_loss_scalar + \
                          self.decoder_loss_scalar + \
                          self.VAE_sample.reg_loss + \
                          self.VAE_feature.reg_loss

        self.losses = [self.total_loss,
                       self.decoder_loss_scalar,
                       self.recon_loss,
                       self.VAE_sample.reg_loss,
                       self.VAE_feature.reg_loss,
                       self.VAE_sample.KL_loss_scalar,
                       self.VAE_feature.KL_loss_scalar]

        self.losses_name = ['total loss',
                            'decoder_loss',
                            'recon_loss',
                            'reg_loss sample',
                            'reg_loss feature',
                            'latent loss sample',
                            'latent loss feature']

        ## Add interpretability regularization terms for latent embedding
        if self.zv_recon_scale is not None:

            self.zv_recon_loss = tf.losses.mean_squared_error(self.VAE_sample.X_target, self.X_mu_zv) * \
                                 int(self.VAE_sample.X_target.shape[-1])

            self.total_loss  += self.zv_recon_loss * self.zv_recon_scale
            self.losses      += [self.zv_recon_loss]
            self.losses_name += ['zv recon loss']

        ## Add interpretability regularization terms for hidden layers in the decoders
        if self.hl_recon_scale is not None:

            for ii,X_mu in enumerate(self.X_mu_hl):

                recon_loss = tf.losses.mean_squared_error(self.VAE_sample.X_target, X_mu) * \
                             int(self.VAE_sample.X_target.shape[-1])

                self.total_loss  += recon_loss * self.hl_recon_scale
                self.losses      += [recon_loss]
                self.losses_name += ['hl-{} recon loss'.format(ii)]

        for loss_name, loss in zip(self.losses_name, self.losses):
            self.summary_list.append(tf.summary.scalar(loss_name, loss))

        ## Optimizer
        # Variables to Train
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name)
        self.theta = self.VAE_sample.theta + self.VAE_feature.theta

        ## Set up optimizer
        with tf.name_scope("global_steps"):

            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            self.decayed_learning_rate = tf.train.exponential_decay(learning_rate = self.learning_rate,
                                                                    global_step   = self.global_step,
                                                                    decay_steps   = self.decay_steps,
                                                                    decay_rate    = self.decay_rate,
                                                                    staircase     = False)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        var_list = self.theta

        logging.info("Build train_op optimizing:")
        logging.info(var_list)

        with tf.name_scope("train_op"):
            self.optimizer = tf.train.AdamOptimizer(self.decayed_learning_rate)

            with tf.control_dependencies(update_ops):
                optimizer = self.optimizer.minimize(loss = self.total_loss,
                                                    global_step = self.global_step,
                                                    var_list = var_list)
                self.train_op = optimizer

        logging.info("train_op:")
        logging.info(self.train_op)

        if self.tensorboard:
            self.merge = tf.summary.merge(self.summary_list)
            self.losses.insert(0,self.merge)


    def start_tensorboard(self, train_writer = None, test_writer = None):
        """ Start tensorboard """

        if train_writer is not None and test_writer is not None:
            self.train_writer = train_writer
            self.test_writer  = test_writer
        else:
            logdir_train = os.path.join(self.logdir_tf,self.name,'train')
            logdir_test  = os.path.join(self.logdir_tf,self.name,'test')
            self.train_writer = tf.summary.FileWriter(logdir_train, self.sess.graph)
            self.test_writer = tf.summary.FileWriter(logdir_test, self.sess.graph)
            logging.info("Starting Tensorboard: {}".format(logdir_train))


    def train(self, sess, dataset = None, initialize = True, train_writer = None, test_writer = None, close_writer = True):
        """ Train the model """

        start_time = time.time()

        self.sess = sess

        # Set random seed
        if initialize:
            tf.set_random_seed(self.random_seed)
            tf.logging.set_verbosity(tf.logging.FATAL)
            sess.run(tf.global_variables_initializer())

        dataset_sample = self.VAE_dict['sample'].data_handler.dataset
        train_data_full, test_data, y_train_full, y_test = dataset_sample
        dataset_feature = self.VAE_dict['feature'].data_handler.dataset

        # Split into validation
        if self.validation_split == 0:
            train_data = train_data_full
            y_train = y_train_full
            validation_data = test_data
            y_validation = y_test
        else:
            len_train = int(train_data.shape[self.permute_axis])
            len_validate = int(len_train * self.validation_split)
            validation_data = train_data_full.swapaxes(0,self.permute_axis)[:len_validate].swapaxes(0,self.permute_axis)
            train_data      = train_data_full.swapaxes(0,self.permute_axis)[len_validate:].swapaxes(0,self.permute_axis)
            y_validation    = y_train.swapaxes(0,self.permute_axis)[:len_validate].swapaxes(0,self.permute_axis)
            y_train         = y_train.swapaxes(0,self.permute_axis)[len_validate:].swapaxes(0,self.permute_axis)

        logging.info("======================== Data Split =============================")
        logging.info("validation_data: " + str(validation_data.shape))
        logging.info("y_validation: " + str(y_validation.shape))
        logging.info("train_data: " + str(train_data.shape))
        logging.info("y_train: " + str(y_train.shape))
        logging.info("test_data: " + str(test_data.shape))
        logging.info("y_test: " + str(y_test.shape))
        logging.info('')

        logging.info("======================== losses =============================")
        logging.info(self.losses)
        logging.info(self.losses_name)

        if self.tensorboard:
            "Starting Tensorboard "
            self.start_tensorboard(train_writer = train_writer, test_writer=  test_writer)

        feed_dict = {}
        feed_dict[self.VAE_feature.X] = dataset_feature[0]

        # Set up for early stopping
        it_tb = -1
        it_test = 0
        it_min = 0
        min_loss = float("inf")
        result = None

        feed_dict_train = {self.VAE_sample.X: train_data,
                           self.VAE_sample.X_target: y_train,
                           self.VAE_feature.X: dataset_feature[0]}

        feed_dict_test = {self.VAE_sample.X: validation_data,
                          self.VAE_sample.X_target: y_validation,
                          self.VAE_feature.X: dataset_feature[0]}

        iter = self.iter

        for VAE_in in self.VAE_dict.values():
            if VAE_in.keep_prob is not None:
                feed_dict[VAE_in.prob] = VAE_in.keep_prob

        ## Set minibatch size
        if self.mb_size <= 1:
            mb_size = int(int(train_data.shape[self.permute_axis]) * self.mb_size)
        else:
            mb_size = int(self.mb_size)

        train_data_size = int(train_data.shape[self.permute_axis])

        # if self.VAE_sample.iterator is None:
        batch = batch_handler(train_data_size, mb_size)

        # Set up batch
        self.solvers = [self.train_op]

        for it in range(iter):
            ## Separate counters for tensorboard and early stopping
            it_tb += 1
            it_test += 1

            idx_out = batch.next_batch()

            if self.VAE_sample.iterator is None:
                feed_dict[self.VAE_sample.X]        = train_data.take(idx_out, axis = -2)
                feed_dict[self.VAE_sample.X_target] = y_train.take(idx_out, axis = -2)


            ## Train a batch
            results = self.sess.run(self.solvers + self.losses, feed_dict = feed_dict)

            # Log train/test results
            if (it_tb % self.log_frequency == 0) or (it_tb == iter - 1):
                logging.info('')
                logging.info('Iter: {}, time: {:.4}s'.format(it, time.time()- start_time))

                ## Batch Loss
                zipped = zip(self.losses_name, results[-len(self.losses_name):])
                format_args = [item for pack in zipped for item in pack]
                str_report = "Batch: {}={:.4}" + ", {}={:.4}" * (len(self.losses_name)-1)
                logging.info(str_report.format(*format_args))

                ## Train Loss
                results = sess.run(self.losses, feed_dict = feed_dict_train)
                results_train = results

                # Write to Summary (writing train data)
                if self.tensorboard:
                    summary = results.pop(0)
                    self.train_writer.add_summary(summary, str(it_tb))

                zipped = zip(self.losses_name,results[-len(self.losses_name):])
                format_args = [item for pack in zipped for item in pack]
                str_report = "Train: {}={:.4}" + ", {}={:.4}" * (len(self.losses_name)-1)
                logging.info(str_report.format(*format_args))

                ## Validation loss

                if self.tensorboard:
                    run_args = {}

                    if self.metadata:
                        run_args["options"] = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_args['run_metadata'] = tf.RunMetadata()

                    results = self.sess.run(self.losses,
                                       feed_dict = feed_dict_test, **run_args)

                    summary = results.pop(0)

                    if self.metadata:
                        self.test_writer.add_run_metadata(run_args["run_metadata"], str(it_tb))

                    self.test_writer.add_summary(summary, str(it_tb))

                    zipped = zip(self.losses_name,results[-len(self.losses_name):])
                    format_args = [item for pack in zipped for item in pack]
                    str_report = "Test: {}={:.4}" + ", {}={:.4}" * (len(self.losses_name)-1)
                    logging.info(str_report.format(*format_args))

                else:
                    results = sess.run(self.losses, feed_dict = feed_dict_test)
                    zipped = zip(self.losses_name,results[-len(self.losses_name):])
                    format_args = [item for pack in zipped for item in pack]
                    str_report = "Test: {}={:.4}" + ", {}={:.4}" * (len(self.losses_name)-1)
                    logging.info(str_report.format(*format_args))

                results_test = results

                if self.early_stopping > 0:
                    logging.info("new min loss: {} vs old {}".format(results[-3], min_loss))
                    delta = min_loss - results[-3]
                    if delta > self.tolerance:
                        it_min = it
                        it_test = 0
                        min_loss =  results[-3]

                    elif it_test > self.early_stopping:
                        break


        if self.tensorboard and close_writer:
            self.train_writer.close()
            self.test_writer.close()

        result = {}

        output = {'model_evaluation': result}
        output['iter'] = it
        output['losses'] = {'train': results_train,
                            'test' : results_test,
                            'name' : self.losses_name}

        #### Calculate layers for full data and save results
        output['split_index'] = self.VAE_dict['sample'].data_handler.split_idx
        output['labels']      = np.array(self.VAE_dict['sample'].data_handler.X.obs['Labels'])

        feed_dict_full = {self.VAE_sample.X  : self.VAE_dict['sample'].data_handler.X.X,
                          self.VAE_sample.X_target : self.VAE_dict['sample'].data_handler.y.X,
                          self.VAE_feature.X : self.VAE_dict['feature'].data_handler.X.X}

        y_full = self.VAE_dict['sample'].data_handler.y.X
        full_data = self.VAE_dict['sample'].data_handler.X.X

        if self.save_recon:

            logging.info("y_full: {}".format(y_full.shape))

            y_reconstruct = np.array(self.reconstruct_X(feed_dict_full))
            y_residual = y_full - y_reconstruct

            y_result = np.array([y_full, y_reconstruct])

            output['reconstruction'] = y_result

        if self.save_LE:

            bottleneck = {}
            for VAE_type, VAE_in in self.VAE_dict.items():
                bottleneck[VAE_type] = np.array(self.calculate_latent_embedding(VAE_in, feed_dict_full))

            output['latent_embedding'] = bottleneck

            bottleneck_var = {}
            for VAE_type, VAE_in in self.VAE_dict.items():
                bottleneck_var[VAE_type] = np.array(self.calculate_latent_variance(VAE_in, feed_dict_full))

            output['latent_embedding_var'] = bottleneck_var

        if self.save_W:
            W = self.get_W(feed_dict_full)
            output['W_mu'] = W
            y_mu = self.VAE_sample.calculate_y_mu(feed_dict_full)
            output['y_mu'] = y_mu

            ## Decoder layers
            decoder_layers_dict = {'sample' : self.VAE_sample.calculate_decoder_layers(feed_dict_full),
                                   'feature': self.VAE_feature.calculate_decoder_layers(feed_dict_full)}
            output['decoder_layers'] = decoder_layers_dict

        return output


    def reconstruct_X(self, feed_dict_in):
        """
        Calculate the X_reconstruct from the mean of the latent variables rather than sampled
        y_reconstruct = [cell, gene]
        """

        y_mu = self.VAE_sample.calculate_y_mu(feed_dict_in)
        W_mu = self.VAE_feature.reconstruct_X(feed_dict_in).transpose()

        feed_dict_Wy = {self.W: W_mu, self.y: y_mu}

        X_reconstruct = np.array(self.sess.run(self.X_dist.mu, feed_dict = feed_dict_Wy))

        return X_reconstruct


    def reconstruct_X_hl(self, feed_dict_in):
        """
        Calculate the X_reconstruct from the mean of the latent variables rather than sampled
        y_reconstruct = [cell, gene]
        """

        z_mu = self.VAE_sample.calculate_z_mu(feed_dict_in)
        v_mu = self.VAE_feature.calculate_z_mu(feed_dict_in)

        feed_dict_Wy = {self.z: z_mu,
                        self.v: v_mu}

        mus = [self.X_mu_zv] + self.X_mu_hl

        X_reconstruct = self.sess.run(mus, feed_dict = feed_dict_Wy)

        return X_reconstruct


    def calculate_X_reconstruct(self, feed_dict_in):
        """ Reconstruct X from input data """

        y_reconstruct = np.array(self.sess.run(self.X_dist.mu, feed_dict = feed_dict_in))

        return y_reconstruct


    def calculate_latent_embedding(self, VAE_in, feed_dict_in):
        """ Calculate bottleneck layer from input data"""
        bottleneck = np.array(self.sess.run(VAE_in.latent_embedding, feed_dict = feed_dict_in))

        return bottleneck


    def calculate_latent_variance(self, VAE_in, feed_dict_in):
        """ Calculate latent variance of VAE_in """
        bottleneck = np.array(self.sess.run(VAE_in.z_var, feed_dict = feed_dict_in))

        return bottleneck


    def calculate_decoder_variance(self, test_data):
        """ Calculate decoder variance """
        var = np.array(self.sess.run(self.X_var, feed_dict = {self.X: test_data}))

        return var


    def get_W(self, feed_dict = {}):
        """ Get W"""
        W = np.array(self.sess.run(self.W, feed_dict=feed_dict))

        return W
