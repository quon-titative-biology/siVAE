import os
import time
import logging

# Tensorflow
import tensorflow as tf
import tensorflow_probability as tfp
tf.compat.v1.disable_v2_behavior()

import numpy as np
import pandas as pd

from .layer import layer
from .layer import KL_divergence_normal
from .layer import create_variance
from .layer import Distribution

from .util import batch_handler

class AutoEncoder(object):

    def __init__(self, logdir_tf = ".", data_handler = None, X_dim = None,
                 h_dims = None, LE_dim = None, architecture = None,
                 iter = 5000, mb_size = 1000, learning_rate = 1e-3, l1_scale = 1e-50, l2_scale = 0, l2_scale_final = None, l1_scale_final = None,
                 early_stopping = 0, tolerance = 0, min_early_stopping = 0,
                 activation_fun = tf.nn.relu, random_seed = 0, log_frequency = 100,
                 batch_norm = False, keep_prob = None,
                 dataAPI = False, tensorboard = False, metadata = False,
                 config = None, validation_split = 0,
                 decay_rate = 1, decay_steps = 1000, save_recon = True,
                 save_LE = True, isVAE = True, beta = 1,
                 decoder_var = 'scalar', save_W = True,
                 name = "", var_dependency = True, optimizer_type = 'adam',
                 kernel_initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                 X_mu_use_bias = True, beta_warmup = 0.5, y_var_type = 'deterministic',
                 classifier = False, attributions_dict = {}, output_distribution = 'normal',
                 log_variational = False, assign_dict={}):

        self.iterator = None
        self.isVAE = isVAE
        self.decoder_var = decoder_var
        self.y_var_type = y_var_type
        self.var_dependency = var_dependency
        self.X_mu_use_bias = X_mu_use_bias
        self.beta_warmup = beta_warmup
        self.beta = beta
        self.output_distribution = output_distribution
        self.log_variational = log_variational

        if not self.isVAE and decoder_var != 'deterministic':
            raise Exception("When isVAE is False, decoder_var must be set to deterministic")

        # NN structure
        self.data_handler           = data_handler
        self.h_dims                 = self.data_handler.h_dims
        self.X_dim                  = self.data_handler.X.shape[-1]
        self.X_target_dim           = self.data_handler.y.shape[-1]
        self.index_latent_embedding = self.data_handler.index_latent_embedding
        shape_X    = list(self.data_handler.X.shape)
        shape_y    = list(self.data_handler.y.shape)
        shape_X[0] = None
        shape_y[0] = None
        self.X_shape        = shape_X
        self.X_target_shape = shape_y

        ## Recalculate h_dims if h_dims = 0
        self.save_LE = save_LE
        self.save_W = save_W
        self.save_recon = save_recon

        ## Convert h_dims
        self.process_h_dims()

        logging.info("h_dims: {}".format(self.h_dims))

        self.name = name

        ## Set activation function
        self.activation_fun = activation_fun

        ## Training Parameters
        self.mb_size = mb_size
        self.iter = iter
        self.keep_prob = keep_prob

        self.kernel_initializer = kernel_initializer
        self.optimizer_type = optimizer_type

        ## Hyperparameters
        self.learning_rate = learning_rate
        self.l2_scale = l2_scale
        if l2_scale_final is None:
            self.l2_scale_final = l2_scale
        else:
            self.l2_scale_final = l2_scale_final
        self.l1_scale = l1_scale
        if l1_scale_final is None:
            self.l1_scale_final = l2_scale
        else:
            self.l1_scale_final = l2_scale_final
        self.random_seed = random_seed

        self.batch_norm = batch_norm

        self.placeholder_reg = {}

        self.feed_dict = {}
        self.logdir_tf = logdir_tf

        self.tensorboard = tensorboard
        self.metadata = metadata
        self.validation_split = validation_split
        self.tolerance = tolerance
        self.summary_list = []

        self.decay_steps = decay_steps
        self.decay_rate = decay_rate


        if self.metadata and not self.tensorboard:
            logging.info("Tensorboard has been switched to true as metadata requires tensorboard")
            self.tensorboard = True

        self.log_frequency = log_frequency
        self.dataAPI = dataAPI
        self.config = config

        self.early_stopping = early_stopping
        self.min_early_stopping = np.max([min_early_stopping, self.beta_warmup])
        self.iter_min_early_stopping = int(self.iter * self.min_early_stopping)

        for attribution,value in attributions_dict.items():
            setattr(self,attribution,value)

    def build_model(self, reset_graph = True, graph = None):
        """ """

        if reset_graph:
            tf.reset_default_graph()

        if graph is None:
            self.Graph = tf.get_default_graph()
        else:
            self.Graph = graph

        tf.set_random_seed(self.random_seed)

        with tf.variable_scope(self.name):
            with tf.name_scope('regularization'):
                self.create_regularization()

        self.hidden_layers = []

        with tf.variable_scope(self.name):

            ## Create placeholders for input X and y
            with tf.name_scope('X'):
                self.X = tf.placeholder(tf.float32, shape=self.X_shape, name = 'X')

            with tf.name_scope('y'):
                self.y = tf.placeholder(tf.float32, shape=self.X_target_shape, name = 'y')

            h_temp = self.X

            ## Create hidden layers
            h_dims = self.h_dims
            for ii, h_dim in enumerate(h_dims):
                logging.info("building layer {} with {} nodes".format(ii,h_dim))
                with tf.variable_scope("hidden_layer_{}".format(ii)):
                    ## Create hidden layers
                    if ii < len(h_dims) - 1:
                        h_temp,_,_ = layer(h = h_temp,
                                           dim = h_dim,
                                           fun = self.activation_fun,
                                           l2_scale = self.l2_scale,
                                           l1_scale = self.l1_scale,
                                           kernel_initializer = self.kernel_initializer,
                                           **self.placeholder_reg)

                        self.hidden_layers.append(h_temp)
                    else:
                        h_temp,_,_ = layer(h = h_temp,
                                           dim = h_dim,
                                           fun = None,
                                           l2_scale = self.l2_scale_final,
                                           l1_scale = self.l1_scale_final,
                                           kernel_initializer = self.kernel_initializer,
                                           **self.placeholder_reg)

                        self.y_pred = h_temp


            ## Create losses
            with tf.name_scope('losses'):
                self.recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.y - self.y_pred),-1))
                tf.losses.add_loss(self.recon_loss)

                self.total_loss = tf.losses.get_total_loss(name = 'total_loss',
                                                           add_regularization_losses = True,
                                                           scope = self.name)

                self.losses = [self.recon_loss, self.total_loss]
                self.losses_name = ['Recon loss', "Total loss"]

                if self.tensorboard:
                    for loss, loss_name in zip(self.losses, self.losses_name):
                        logging.info("Adding {} to tensorboard".format(loss_name))
                        self.summary_list.append(tf.summary.scalar(loss_name, loss))


            self.theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)

            ## Create optimizer
            with tf.name_scope("global_steps"):

                self.global_step = tf.Variable(0, name='global_step', trainable=False)

                self.decayed_learning_rate = tf.train.exponential_decay(learning_rate = self.learning_rate,
                                                                        global_step   = self.global_step,
                                                                        decay_steps   = self.decay_steps,
                                                                        decay_rate    = self.decay_rate,
                                                                        staircase     = False)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = self.name)

            with tf.name_scope("train_op"):

                if self.optimizer_type == 'adam':
                    optimizer = tf.train.AdamOptimizer
                elif self.optimizer_type == 'grad':
                    optimizer = tf.compat.v1.train.GradientDescentOptimizer

                self.optimizer = optimizer(self.decayed_learning_rate)

                with tf.control_dependencies(update_ops):
                    optimizer = self.optimizer.minimize(loss = self.total_loss,
                                                        global_step = self.global_step,
                                                        var_list = self.theta)
                    self.train_op = optimizer

            if self.tensorboard:
                self.merge = tf.summary.merge(self.summary_list)
                self.losses.insert(0,self.merge)


    def train(self, sess, dataset = None, initialize = True, close_writer = True,
              train_writer = None, test_writer = None, initial_iter = 0):

        logging.info('Start training')

        start_time = time.time()

        self.sess = sess

        # Set random seed
        if initialize:
            with tf.name_scope('initialize'):
                tf.set_random_seed(self.random_seed)
                tf.logging.set_verbosity(tf.logging.FATAL)
                sess.run(tf.global_variables_initializer())

        if dataset is None:
            dataset = self.data_handler.dataset

        train_data_full, test_data, y_train_full, y_test = dataset

        # Split into validation
        if self.validation_split == 0:
            train_data = train_data_full
            y_train = y_train_full
            validation_data = test_data
            y_validation = y_test
        else:
            len_train = int(train_data.shape[0])
            len_validate = int(len_train * self.validation_split)
            validation_data = train_data_full[:len_validate]
            train_data      = train_data_full[len_validate:]
            y_validation    = y_train[:len_validate]
            y_train         = y_train[len_validate:]

        logging.info("======================== Data Split =============================")
        logging.info("validation_data: " + str(validation_data.shape))
        logging.info("y_validation: " + str(y_validation.shape))
        logging.info("train_data: " + str(train_data.shape))
        logging.info("y_train: " + str(y_train.shape))
        logging.info("test_data: " + str(test_data.shape))
        logging.info("y_test: " + str(y_test.shape))

        if self.tensorboard:
            "Starting Tensorboard "
            self.start_tensorboard(train_writer = train_writer, test_writer=  test_writer)

        feed_dict             = self.initialize_feed_dict()

        feed_dict_train       = self.initialize_feed_dict({self.X: train_data,
                                                           self.y: y_train})

        feed_dict_test        = self.initialize_feed_dict({self.X: test_data,
                                                           self.y: y_test})

        feed_dict_validation  = self.initialize_feed_dict({self.X: validation_data,
                                                           self.y: y_validation})


        self.feed_dict_test       = feed_dict_test
        self.feed_dict_train      = feed_dict_train
        self.feed_dict_validation = feed_dict_validation

        if self.batch_norm:
            feed_dict[self.is_train] = True


        # Set up for early stopping
        it_tb = -1 + initial_iter
        it_test = 0
        it_min = 0
        min_loss = float("inf")
        result = None

        iter = self.iter
        if self.keep_prob is not None:
            feed_dict[self.prob] = self.keep_prob

        ## Set up minibatch
        # Set minibatch size
        if self.mb_size <= 1:
            mb_size = int(int(train_data.shape[0]) * self.mb_size)
        else:
            mb_size = int(self.mb_size)

        train_data_size = int(train_data.shape[0])
        epoch = int((iter * mb_size) / train_data_size) + 1

        # Set up batch
        self.solvers = [self.train_op]

        if self.dataAPI:
            logging.info('Using dataAPI')
            self.dataset = tf.data.Dataset.from_tensor_slices(tuple(self.dataAPI_placeholders)).shuffle(train_data_size).repeat(epoch).batch(mb_size)
            self.initializer = self.iterator.make_initializer(self.dataset)
            sess.run(self.initializer, feed_dict = {self.dataAPI_placeholders[0]: train_data,
                                                    self.dataAPI_placeholders[1]: y_train})

        elif self.data_handler.iterator is not None:
            logging.info('Using data_handler iterator')

        else:
            logging.info('Using batch')
            batch = batch_handler(train_data_size, mb_size)

        for it in range(iter):
            it_tb += 1
            it_test += 1

            if self.dataAPI:
                pass
            elif self.data_handler.iterator is not None:
                pass
            else:
                idx_out = batch.next_batch()
                train_data.take(idx_out, axis = -2)
                feed_dict[self.X] = train_data.take(idx_out, axis = -2)
                feed_dict[self.y] =  y_train.take(idx_out, axis = -2)

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
                results_train = results[-len(self.losses_name):]
                # Write to Summary (writing train data)
                if self.tensorboard:
                    summary = results.pop(0)
                    self.train_writer.add_summary(summary, str(it_tb))

                zipped = zip(self.losses_name,results[-len(self.losses_name):])
                format_args = [item for pack in zipped for item in pack]
                str_report = "Train: {}={:.4}" + ", {}={:.4}" * (len(self.losses_name)-1)
                logging.info(str_report.format(*format_args))

                ## Train Loss
                results = sess.run(self.losses, feed_dict = feed_dict_test)
                results_test = results[-len(self.losses_name):]

                zipped = zip(self.losses_name,results[-len(self.losses_name):])
                format_args = [item for pack in zipped for item in pack]
                str_report = "Test: {}={:.4}" + ", {}={:.4}" * (len(self.losses_name)-1)
                logging.info(str_report.format(*format_args))

                ## Validation loss
                if self.tensorboard:
                    run_args = {}
                    if self.metadata:
                        run_args["options"] = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_args['run_metadata'] = tf.RunMetadata()
                    results = self.sess.run(self.losses, feed_dict = feed_dict_validation, **run_args)
                    summary = results.pop(0)
                    if self.metadata:
                        self.test_writer.add_run_metadata(run_args["run_metadata"], str(it_tb))
                    self.test_writer.add_summary(summary, str(it_tb))

                    zipped = zip(self.losses_name,results[-len(self.losses_name):])
                    format_args = [item for pack in zipped for item in pack]
                    str_report = "Validation: {}={:.4}" + ", {}={:.4}" * (len(self.losses_name)-1)
                    logging.info(str_report.format(*format_args))

                else:
                    results = sess.run(self.losses, feed_dict = feed_dict_validation)
                    zipped = zip(self.losses_name,results[-len(self.losses_name):])
                    format_args = [item for pack in zipped for item in pack]
                    str_report = "Validation: {}={:.4}" + ", {}={:.4}" * (len(self.losses_name)-1)
                    logging.info(str_report.format(*format_args))

                if self.early_stopping > 0:
                    logging.info("new min loss: {} vs old {}".format(results[-3], min_loss))
                    delta = min_loss - results[-3]
                    if delta > self.tolerance:
                        it_min = it
                        it_test = 0
                        min_loss =  results[-3]

                    elif it_test > self.early_stopping and it_tb > self.iter_min_early_stopping:
                        break

        if result is None:

            result = {}

            logging.info('y_test shape: {}'.format(y_test.shape))
            logging.info('feed_dict_test: {}'.format(feed_dict_test))

        output = {'model_evaluation': result}
        output['losses'] = {'train': results_train,
                            'test' : results_test,
                            'name' : self.losses_name}

        if self.tensorboard:
            logging.info("Reults saved to tensorboard")

        #### Calculate layers for full data and save results
        output['split_index'] = self.data_handler.split_idx
        output['labels']      = np.array(self.data_handler.X.obs_names)

        full_data = self.data_handler.X.X
        feed_dict_full = self.initialize_feed_dict({self.X: full_data})
        y_full = self.data_handler.y.X

        if self.save_recon:

            logging.info("y_full: {}".format(y_full.shape))

            y_reconstruct = np.array(self.sess.run(self.y_pred,feed_dict = feed_dict_full))
            y_residual = y_full - y_reconstruct
            y_result = np.array([y_full, y_reconstruct, y_residual])

            logging.info("y_result.shape = {}".format(y_result.shape))

            output['reconstruction'] = y_result

        return output

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


    def initialize_feed_dict(self, feed_dict = {}):
        """ Create a new feed_dict from input feed_dict and self.feed_dict """
        feed_dict_new = {}
        feed_dict_new.update(self.feed_dict)
        feed_dict_new.update(feed_dict)
        return feed_dict_new


    def process_h_dims(self):
        """ Process number of nodes for hidden layers """

        if len(self.h_dims) != 0:

            if np.any(np.array(self.h_dims[:-1]) == 0):

                h_dims_old = [self.X_dim] + self.h_dims
                h_dims_new = h_dims_old
                logging.info("Converting h_dims: ".format(h_dims_old))

                for ii, h_dim in enumerate(h_dims_old[:-1]):

                    if h_dim == 0:
                        logging.info(ii)
                        end = ii + 1

                        if h_dims_old[end] != 0:
                            new_values = np.linspace(h_dims_old[start], h_dims_old[end], end - start + 1).astype(int)
                            h_dims_new[start+1:end] = new_values[1:-1]

                    else:
                        start = ii

                logging.info("h_dims_new: ".format(h_dims_new))
                self.h_dims = h_dims_new[1:]

            self.h_dims[-1] = self.X_target_dim


    def create_regularization(self):
        """ Create regularizations for the model """

        if self.batch_norm:
            self.is_train = tf.placeholder_with_default(False, shape=(), name = "is_train")
            self.placeholder_reg["batch_norm"] = self.is_train

        if self.keep_prob is not None:
            self.prob = tf.placeholder_with_default(np.array(1).astype("float32"),
                                                    shape=(), name = 'prob')
            self.placeholder_reg["drop"] = self.prob

        self.do_sample = tf.placeholder_with_default(True, shape=(), name = "do_sample")
