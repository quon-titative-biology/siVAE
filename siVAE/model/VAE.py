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
                 iter = 5000, mb_size = 1000, learning_rate = 1e-3, l1_scale = 1e-50, l2_scale = 0,
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
                 classifier = False, attributions_dict = {}):

        self.iterator = None
        self.isVAE = isVAE
        self.decoder_var = decoder_var
        self.y_var_type = y_var_type
        self.var_dependency = var_dependency
        self.X_mu_use_bias = X_mu_use_bias
        self.beta_warmup = beta_warmup
        self.beta = beta
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
        self.l1_scale = l1_scale
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

            with tf.variable_scope('feed'):
                self.create_feed(dataAPI = self.dataAPI)

            with tf.variable_scope('model'):
                self.build_basic_network()


    def create_feed(self, dataAPI = False, batch = None, epoch = None, shuffle = None):
        """ y = X_reconstructed for variational autoencoder """

        if self.data_handler.iterator is not None:
            X = self.data_handler.iterator.get_next()[0]
            self.X = X
            self.X_target = X
        else:
            if not hasattr(self, 'X'):
                with tf.name_scope('X_placeholder'):
                    self.X = tf.placeholder(tf.float32, shape=self.X_shape, name = 'X')

            if not hasattr(self, 'X_target'):
                with tf.name_scope('X_target_placeholder'):
                    self.X_target = tf.placeholder(tf.float32, shape=self.X_target_shape, name = 'X_target')

        if dataAPI:
            self.dataAPI_placeholders = [self.X,self.X_target]

            ## dummy_dataset created as placeholder
            dummy_dataset = tf.compat.v1.data.Dataset.from_tensor_slices((self.X,self.X_target)).batch(10000)

            iterator = tf.compat.v1.data.Iterator.from_structure(dummy_dataset.output_types,
                                                                 dummy_dataset.output_shapes)

            self.X, self.X_target = iterator.get_next()

            self.iterator = iterator

        if not hasattr(self, 'initial_feed'):
            self.initial_feed = self.X


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


    def create_decoder_loss(self):
        """ Create decoder loss """

        with tf.name_scope('mean_squared_error'):
            self.recon_loss_ae = self.mean_squared_error(self.X_mu, self.X_target)

        with tf.name_scope('log_likelihood'):
            if self.isVAE and self.decoder_var != 'deterministic':
                self.decoder_loss_old = self.X_logvar / 2 + self.recon_loss_ae / self.X_var / 2 + tf.constant(0.5 * np.log(2. * np.pi), dtype = self.X_var.dtype)
                self.decoder_loss_mse    = tf.square(self.X_mu - self.X_target)
                self.decoder_loss_mse_scaled = self.recon_loss_ae / self.X_var / 2
                self.decoder_loss_logvar = self.decoder_loss_mse * 0 + self.X_logvar / 2
                self.decoder_loss = -self.X_dist.dist.log_prob(self.X_target)
            else:
                self.decoder_loss = tf.reduce_sum(self.recon_loss_ae,-1)


    def build_encoder(self, h_temp, h_dims, kernel_initializer, variable_scope = 'Encoder', isVAE = False):
        """ Build encoder """

        logging.info("Building Encoder")

        l2_scale_encoder = self.l2_scale
        l1_scale_encoder = self.l1_scale

        with tf.variable_scope(variable_scope):
            for ii, h_dim in enumerate(h_dims):
                logging.info("building layer {} with {} nodes".format(ii,h_dim))
                with tf.variable_scope("hidden_layer_{}".format(ii)):
                    if ii < len(h_dims) - 1:
                        ## Create hidden layers
                        h_temp,_,_ = layer(h = h_temp,
                                           dim = h_dim,
                                           fun = self.activation_fun,
                                           l2_scale = l2_scale_encoder,
                                           l1_scale = l1_scale_encoder,
                                           kernel_initializer = kernel_initializer,
                                           **self.placeholder_reg)
                    else:

                        ## create latent embedding layer
                        if isVAE:
                            var_type = 'diagonal'
                        else:
                            var_type = 'deterministic'

                        prob_layer = Distribution(h = h_temp,
                                                  dim = h_dim,
                                                  fun = None,
                                                  l2_scale = l2_scale_encoder,
                                                  l1_scale = l1_scale_encoder,
                                                  var_type = 'diagonal')

                self.hidden_layers.append(h_temp)

        return prob_layer


    def build_decoder(self, h_dims, h_temp, l1_scale, l2_scale, kernel_initializer, variable_scope = 'decoder'):

        logging.info("Building " + variable_scope + "with dims {}".format(h_dims))

        self.decoder_layers = []

        with tf.variable_scope(variable_scope):

            for ii, h_dim in enumerate(h_dims):

                with tf.variable_scope("hidden_layer_{}".format(ii)):

                    if ii < len(h_dims) - 2:
                        ## Hidden layers
                        fun = self.activation_fun

                        h_temp,_,_ = layer(h = h_temp, dim = h_dim, fun = fun,
                                           l2_scale = l2_scale, l1_scale = l1_scale,
                                           kernel_initializer = kernel_initializer)

                    elif ii == len(h_dims) - 2:
                        ## Last layer before final layer

                        prob_layer  = Distribution(h   = h_temp,
                                                   dim = h_dim,
                                                   fun = None,
                                                   var_type = self.y_var_type,
                                                   l2_scale = l2_scale,
                                                   l1_scale = l1_scale)

                        self.y_dist = prob_layer
                        self.y_sample, self.y_mu, self.y_var, self.y_logvar, self.y_eps = prob_layer.get_tensors()
                        h_temp = self.y_sample

                    elif ii == len(h_dims) - 1:

                        ## If there is no y layer, set y to z
                        if len(h_dims) == 1:
                            self.y_dist = self.z_dist
                            self.y_sample = self.z_sample
                            self.y_mu = self.z_mu
                            self.y_var = self.z_var
                            self.y_logvar = self.z_logvar
                            self.y_eps    = self.z_eps

                        with tf.variable_scope('output', reuse = tf.AUTO_REUSE):

                            self.output_variable_scope = tf.get_variable_scope().name

                            prob_layer,W,b = self.build_output_layer(h_temp, h_dim = h_dim,
                                                                     fun = self.activation_fun,
                                                                     var_type = self.decoder_var,
                                                                     var_dependency = self.var_dependency,
                                                                     l2_scale = l2_scale,
                                                                     l1_scale = l1_scale,
                                                                     use_bias = self.X_mu_use_bias)

                        self.X_dist = prob_layer
                        self.X_sample, self.X_mu, self.X_var, self.X_logvar, self.X_eps = prob_layer.get_tensors()
                        self.X_sample = tf.cond(self.do_sample, lambda: self.X_sample, lambda: self.X_mu, 'X_sample_cond')
                        self.W = W
                        self.b = b
                        h_temp = self.X_mu

                self.decoder_layers.append(h_temp)
                self.hidden_layers.append(h_temp)
                self.summary_list.append(tf.summary.histogram("hidden_layer_{}", h_temp))

        return prob_layer


    def build_output_layer(self, h_temp, fun, var_type, var_dependency, l2_scale, l1_scale, use_bias,
                           h_dim = None, W = None, b = None, h_mu = None, h_std = None, custom = False, **kwargs):
        """"""

        h_mu, W, b = layer(h_temp, h_dim, fun = None,
                           name = "{}_{}".format('X','mu'),
                           l1_scale = l1_scale, l2_scale = l2_scale,
                           custom = custom, use_bias = use_bias,
                           W = W, b = b, **kwargs)


        with tf.variable_scope('Distribution'):
            prob_layer  = Distribution(h = h_temp,
                                       h_mu = h_mu,
                                       h_std = h_std,
                                       dim = h_dim,
                                       fun = None,
                                       var_type = var_type,
                                       var_dependency = var_dependency,
                                       l2_scale = l2_scale,
                                       l1_scale = l1_scale,
                                       **kwargs)

        return prob_layer, W, b


    def mean_squared_error(self, target, reconstruction):
        """ calculate mean squared error between target and reconstruction """
        return tf.square(target - reconstruction)


    def build_basic_network(self):
        """ Build network """
        logging.info("Building Basic Network")

        with tf.variable_scope('NN'):
            self.X_out = self.predict_X(self.initial_feed)

        with tf.name_scope('losses'):

            ## Autoencoder
            logging.info("recon_loss_ae")
            self.create_decoder_loss()
            logging.info(self.recon_loss_ae)
            logging.info("decoder_loss")
            logging.info(self.decoder_loss)

            # VAE
            self.create_loss()


    def create_loss(self):
        """ """
        logging.info("Creating loss")
        logging.info("GraphKeys Losses")
        logging.info(tf.get_collection(tf.GraphKeys.LOSSES))

        ## Set up optimizer
        with tf.name_scope("global_steps"):


            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            self.decayed_learning_rate= tf.train.exponential_decay(learning_rate = self.learning_rate,
                                                                   global_step   = self.global_step,
                                                                   decay_steps   = self.decay_steps,
                                                                   decay_rate    = self.decay_rate,
                                                                   staircase     = False)
            if self.isVAE:
                with tf.name_scope("beta_scaled"):

                    self.iter_beta_full = self.iter * self.beta_warmup

                    if self.beta_warmup != 0:
                        self.beta_scale = tf.clip_by_value(tf.cast(self.global_step, dtype = tf.float32) / (self.iter_beta_full), 0, 1)
                    else:
                        self.beta_scale = 1

                    self.beta_scaled = self.beta * self.beta_scale
                    self.summary_list.append(tf.summary.scalar('beta_scaled', self.beta_scaled))


        if self.isVAE:
            logging.info("KL_loss_ae")
            logging.info("beta: {}".format(self.beta))
            # self.KL_loss = self.create_KL_loss(beta = self.beta)
            self.KL_loss = self.beta_scaled * tfp.distributions.kl_divergence(self.z_dist.dist,
                                                                           self.z_prior,
                                                                           allow_nan_stats = False,
                                                                           name = 'z_KL_divergence')

            logging.info(self.KL_loss)

        ## l2_loss
        self.reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses(scope = self.name))
        logging.info(" ")
        logging.info("reg_loss: {}".format(self.reg_loss.shape))
        logging.info(self.reg_loss)
        logging.info(tf.losses.get_regularization_losses(scope = self.name))

        ## recon_loss
        self.recon_loss_scalar = tf.reduce_mean(tf.reduce_sum(self.recon_loss_ae,-1))
        logging.info("recon_loss_scalar: {}".format(self.recon_loss_scalar.shape))
        logging.info(self.recon_loss_scalar)
        logging.info(self.recon_loss_ae)

        ## decoder_loss
        self.decoder_loss_scalar = tf.reduce_mean(self.decoder_loss)
        tf.losses.add_loss(self.decoder_loss_scalar)
        logging.info("decoder_loss_scalar: {}".format(self.decoder_loss_scalar.shape))
        logging.info(self.decoder_loss_scalar)
        logging.info(self.decoder_loss)

        ## KL_loss
        if self.isVAE:
            self.KL_loss_scalar = tf.reduce_mean(self.KL_loss)
            tf.losses.add_loss(self.KL_loss_scalar)
            logging.info("KL_loss: {}".format(self.KL_loss_scalar.shape))
            logging.info(self.KL_loss_scalar)
            logging.info(self.KL_loss)

        ## total_loss
        self.total_loss = tf.losses.get_total_loss(name = 'total_loss',
                                                   add_regularization_losses = True,
                                                   scope = self.name)
        logging.info("total_loss: {}".format(self.total_loss.shape))
        logging.info(self.total_loss)

        ## Log all to tensorboard
        self.losses = [self.total_loss, self.reg_loss, self.recon_loss_scalar, self.decoder_loss_scalar]
        self.losses_name = ['Total loss', "reg loss", "recon loss", "decoder loss"]

        if self.isVAE:
            self.losses += [self.KL_loss_scalar]
            self.losses_name += ['KL_loss']

        if self.tensorboard:
            for loss, loss_name in zip(self.losses, self.losses_name):
                logging.info("Adding {} to tensorboard".format(loss_name))
                self.summary_list.append(tf.summary.scalar(loss_name, loss))

        self.theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)
        self.theta_encoder = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = os.path.join(self.name,"NN/masked_X/Encoder"))
        self.theta_decoder = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = os.path.join(self.name,"NN/masked_X/Decoder"))

        logging.info("thetas")
        logging.info(self.theta)
        logging.info("theta_encoder")
        logging.info(self.theta_encoder)
        logging.info("theta_decoder")
        logging.info(self.theta_decoder)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = self.name)

        logging.info("Build train_op")

        with tf.name_scope("train_op"):

            if self.optimizer_type == 'adam':
                self.optimizer = tf.train.AdamOptimizer(self.decayed_learning_rate)

            with tf.control_dependencies(update_ops):
                optimizer = self.optimizer.minimize(loss = self.total_loss,
                                                    global_step = self.global_step,
                                                    var_list = self.theta)
                self.train_op = optimizer

        logging.info("train_op:")
        logging.info(self.train_op)

        if self.tensorboard:
            self.merge = tf.summary.merge(self.summary_list)
            self.losses.insert(0,self.merge)


    def predict_X(self, h_temp):
        """ Build network that takes in masked X then returns Y """

        self.hidden_layers = []

        h_dims_encoder = self.h_dims[:self.index_latent_embedding + 1]
        self.h_dims_encoder = h_dims_encoder
        h_dims_decoder = self.h_dims[self.index_latent_embedding + 1:]
        self.h_dims_decoder = h_dims_decoder

        ## Encoder
        prob_layer = self.build_encoder(h_temp = h_temp, h_dims = h_dims_encoder,
                                        kernel_initializer = self.kernel_initializer,
                                        variable_scope = 'Encoder', isVAE = self.isVAE)

        self.z_dist = prob_layer
        self.z_sample, self.z_mu, self.z_var, self.z_logvar, self.z_eps = prob_layer.get_tensors()
        self.z_sample = tf.cond(self.do_sample, lambda: self.z_sample, lambda: self.z_mu, 'z_sample_cond')
        self.z_prior = tfp.distributions.MultivariateNormalDiag(loc = tf.zeros(int(self.z_sample.shape[-1])),
                                                                scale_diag = tf.ones(int(self.z_sample.shape[-1])))

        self.latent_embedding = self.z_mu

        if self.tensorboard:
            with tf.name_scope("z"):
                self.summary_list.append(tf.summary.histogram("z_sample", self.z_sample[0]))
                if self.isVAE:
                    self.summary_list.append(tf.summary.histogram("z_mu", self.z_mu[0]))
                    self.summary_list.append(tf.summary.histogram("z_var", self.z_var[0]))
                    self.summary_list.append(tf.summary.histogram("z_logvar", self.z_logvar[0]))

        ## Decoder
        prob_layer = self.build_decoder(h_temp = self.z_sample,
                                        h_dims = h_dims_decoder,
                                        variable_scope = "Decoder",
                                        kernel_initializer = self.kernel_initializer,
                                        l2_scale = self.l2_scale,
                                        l1_scale = self.l1_scale)

        if self.tensorboard:
            with tf.name_scope("X"):
                self.summary_list.append(tf.summary.histogram("X_mu", self.X_mu[0]))
                if self.isVAE and self.decoder_var != 'deterministic':
                    self.summary_list.append(tf.summary.histogram("X_var", self.X_var))
                    self.summary_list.append(tf.summary.histogram("X_logvar", self.X_logvar))
                    self.summary_list.append(tf.summary.histogram("X_eps", self.X_eps[0]))

        return self.X_sample


    def initialize_feed_dict(self, feed_dict = {}):
        """ Create a new feed_dict from input feed_dict and self.feed_dict """
        feed_dict_new = {}
        feed_dict_new.update(self.feed_dict)
        feed_dict_new.update(feed_dict)
        return feed_dict_new


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
        logging.info(np.min(np.std(y_train,0)))
        logging.info("test_data: " + str(test_data.shape))
        logging.info("y_test: " + str(y_test.shape))
        logging.info(np.min(np.std(y_test,0)))

        if self.tensorboard:
            "Starting Tensorboard "
            self.start_tensorboard(train_writer = train_writer, test_writer=  test_writer)

        feed_dict             = self.initialize_feed_dict()

        feed_dict_train       = self.initialize_feed_dict({self.X: train_data,
                                                           self.X_target: y_train})

        feed_dict_test        = self.initialize_feed_dict({self.X: test_data,
                                                           self.X_target: y_test})

        feed_dict_validation  = self.initialize_feed_dict({self.X: validation_data,
                                                           self.X_target: y_validation})

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

        # Set up batch
        self.solvers = [self.train_op]

        for it in range(iter):
            it_tb += 1
            it_test += 1

            if self.dataAPI:
                pass
            elif self.data_handler.iterator is not None:
                pass
            else:
                idx_out = batch.next_batch()
                feed_dict[self.X] = train_data.take(idx_out, axis = -2)
                feed_dict[self.X_target] =  y_train.take(idx_out, axis = -2)

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

            y_reconstruct = np.array(self.reconstruct_X(feed_dict_full))
            y_residual = y_full - y_reconstruct
            y_result = np.array([y_full, y_reconstruct, y_residual])

            logging.info("y_result.shape = {}".format(y_result.shape))

            output['reconstruction'] = y_result

        if self.save_LE:

            if np.all(y_test == y_train):
                y_full = y_train
                full_data = train_data
            else:
                y_full = np.concatenate([y_test, y_train], axis = -2)
                full_data = np.concatenate([test_data, train_data], axis = -2)

            z_mu = np.array(self.calculate_z_mu(feed_dict_full))
            logging.info("Saving bottleneck layer")
            logging.info("=======================")
            logging.info("bottleneck.shape: {}".format(z_mu.shape))
            output['z_mu'] = z_mu

            y = np.array(self.calculate_y_mu(feed_dict_full))
            logging.info("Saving y layer")
            logging.info("=======================")
            logging.info("y_mu.shape: {}".format(y.shape))
            output['y_mu'] = y

            decoder_layers = self.calculate_decoder_layers(feed_dict_full)
            logging.info("Saving decoder layers")
            output['decoder_layers'] = decoder_layers

            if self.isVAE and self.decoder_var != 'deterministic':
                z_var = np.array(self.calculate_latent_variance(feed_dict_full))
                logging.info("Saving z_var")
                logging.info("=======================")
                logging.info("self.z_logvar: {}".format(z_var.shape))
                output['z_var'] = z_var

                X_var = np.array(self.calculate_decoder_variance(feed_dict_full))
                logging.info("Saving X_var")
                logging.info("=======================")
                logging.info("self.X_var: {}".format(X_var.shape))
                output['X_var'] = X_var

        if self.save_W:
            W = self.get_W()
            output['W'] = W

        if self.tensorboard and close_writer:
            self.train_writer.close()
            self.test_writer.close()

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


    def calculate_X_reconstruct(self, feed_dict_test):
        """ Calcualte y from input data """

        y_reconstruct = np.array(self.sess.run([self.X_out], feed_dict = feed_dict_test))

        logging.info("y_reconstruct shape: {}".format(np.array(y_reconstruct).shape))

        return y_reconstruct


    def reconstruct_X(self, feed_dict_in):
        """
        Calculate the X_reconstruct from the mean of the latent variables rather than sampled
        """

        y_mu = self.calculate_y_mu(feed_dict_in)

        feed_dict_Wy = self.initialize_feed_dict({self.y_sample: y_mu})

        feed_dict_Wy = {self.y_sample: y_mu}

        X_reconstruct = np.array(self.sess.run(self.X_dist.mu, feed_dict = feed_dict_Wy))

        logging.info("X_reconstruct shape: {}".format(np.array(X_reconstruct).shape))

        return X_reconstruct


    def calculate_y_sample(self, feed_dict_test):
        """
        Calcualte y from input data and return matrix with dimensions [n_cell x n_gene]
        """

        y_reconstruct = np.array(self.sess.run(self.y_sample, feed_dict = feed_dict_test))

        logging.info("y_reconstruct shape: {}".format(np.array(y_reconstruct).shape))

        return y_reconstruct


    def calculate_z(self, feed_dict_input, sample = False):
        """ Calculate predicted mean of latent variable z from input data"""

        if sample:
            z = np.array(self.sess.run(self.z_sample, feed_dict = feed_dict_input))
        else:
            z = np.array(self.sess.run(self.z_mu, feed_dict = feed_dict_input))
        logging.info("z shape: {}".format(np.array(z).shape))

        return z


    def calculate_z_mu(self, feed_dict_input):
        """ Calculate predicted mean of latent variable z from input data"""

        z_mu = np.array(self.sess.run(self.latent_embedding, feed_dict = feed_dict_input))

        logging.info("z_mu shape: {}".format(np.array(z_mu).shape))

        return z_mu


    def calculate_y_mu(self, feed_dict_input):
        """ Calculate predicted mean of latent variable y from input data without sampling z """

        z_mu = self.calculate_z_mu(feed_dict_input)

        if self.z_mu == self.y_mu:
            y_mu = z_mu
        else:
            y_mu = np.array(self.sess.run(self.y_mu, feed_dict = {self.z_sample: z_mu}))

        logging.info("y_mu: {}".format(np.array(y_mu).shape))

        return y_mu


    def calculate_decoder_layers(self, feed_dict_input, sample_z = False):
        """ Calculate predicted mean of latent variable y from input data without sampling z """

        z = self.calculate_z(feed_dict_input, sample = sample_z)

        hl_list = self.sess.run(self.decoder_layers, feed_dict = {self.z_sample: z})

        return hl_list


    def calculate_latent_variance(self, feed_dict_test):
        """ Calculate bottleneck layer from input data"""
        bottleneck = np.array(self.sess.run(self.z_var, feed_dict = feed_dict_test))

        logging.info("bottleneck shape: {}".format(np.array(bottleneck).shape))

        return bottleneck


    def calculate_decoder_variance(self, feed_dict_test):
        """ Calculate bottleneck layer from input data"""
        var = np.array(self.sess.run(self.X_var, feed_dict = feed_dict_test))

        logging.info("decoder variance shape: {}".format(np.array(var).shape))

        return var


    def get_W(self):
        """ Get W """
        W = np.array(self.sess.run(self.W))

        logging.info("W shape: {}".format(W.shape))

        return W


    def update_feed_dict(self, feed_dict):
        """ Update the basic feed_dict """
        self.feed_dict.update(feed_dict)
