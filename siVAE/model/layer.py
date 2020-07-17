import os
import time
import logging

# Tensorflow
import tensorflow as tf
import tensorflow_probability as tfp

from .util import scope_name

def layer(h, dim = None, fun = None, name = None, l2_scale = 1e-50, l1_scale = 1e-50,
          drop = None,  batch_norm = None, custom = False, W = None, b = None, probability = False,
          kernel_initializer = tf.contrib.layers.xavier_initializer(uniform = False),
          bias_initializer = tf.zeros_initializer(), use_bias = True, graph = None, axis_batch = -2, match_dim = False, reuse = False):
    """ defines a layer using tf.layers """

    variable_scope = 'layer'

    if not reuse:
        variable_scope = scope_name(variable_scope)

    if l2_scale >= l1_scale:
        regularizer = tf.contrib.layers.l2_regularizer(scale = l2_scale)
    else:
        regularizer = tf.contrib.layers.l1_regularizer(scale = l1_scale)

    with tf.variable_scope(variable_scope):

        if batch_norm is not None:
            h = tf.layers.batch_normalization(h, training=batch_norm)

        if not custom:
            print ('not custom')

            if W is not None:
                raise Exception('Warning: W is not being used with tf.layer.dense, set custom to True')

            if len(h.shape) != 2 and match_dim:
                raise Exception('Tensor with more than 2 dimensions is being used on tf.layer.dense')

            h = tf.layers.dense(inputs = h, units = dim, activation = fun,
                                kernel_initializer = kernel_initializer,
                                kernel_regularizer = regularizer,
                                use_bias = use_bias,
                                bias_initializer = bias_initializer,
                                name = name)

            W = tf.get_default_graph().get_tensor_by_name(os.path.split(h.name)[0] + '/kernel:0')

            if use_bias:
                if b is not None:
                    raise Exception('Warning: b is not being used with tf.layer.dense, set custom to True')
                b = tf.get_default_graph().get_tensor_by_name(os.path.split(h.name)[0] + '/bias:0')

        else:
            logging.info('custom')
            if W is None or (b is None and use_bias):
                if match_dim:
                    shape = [str(d) for d in h.shape]
                    shape.pop(axis_batch)
                    shape.append(dim)
                    shape = [int(d) for d in shape]
                    shape_W = list(shape)
                    shape_b = [shape[0],shape[-1]]
                else:
                    shape_W = [int(h.shape[-1]),dim]
                    shape_b = [dim]

            kwargs = {'dtype'      : tf.float32,
                      'trainable'  : True
            }

            variable_scope = scope_name('dense')

            with tf.variable_scope('dense'):

                if W is None:

                    W = tf.get_variable(name = 'kernel',
                                        shape = shape_W,
                                        initializer = kernel_initializer,
                                        regularizer = regularizer,
                                        **kwargs)

                if use_bias:

                    if b is None:

                        b = tf.get_variable(name = 'bias',
                                            shape = shape_b,
                                            initializer = bias_initializer,
                                            regularizer = regularizer,
                                            **kwargs)
            if axis_batch == 0:
                h = tf.transpose(h,[1,0,2])
                h = tf.matmul(h,W)
                h = tf.transpose(h,[1,0,2])
            else:
                h = tf.matmul(h,W)
            if use_bias:
                h = h + b

            if fun is not None:
                h = fun(h)

        if drop is not None:
            h = tf.nn.dropout(h, drop)

    return(h, W, b)


def KL_divergence_normal(dist1, dist2 = None):
    """
    KL Divergence
    D_KL[N(mu_1,exp(logvar_1)I) || N(mu_2,exp(logvar_2)I)]
    """

    mu_1, var_1, logvar_1 = dist1

    if dist2 is None:

        loss = 0.5 * (mu_1**2 + var_1 - 1 - logvar_1)

    else:
        mu_2, var_2, logvar_2 = dist2
        loss = 0.5 * ((logvar2 - logvar_1)
                      - 1
                      + var_1 / var_2
                      + (mu_2 - mu_1)**2 / var_2
                     )

    loss = tf.identity(loss, "KL_Divergence_Normal")

    return loss


def create_variance(h, var_dependency, var_type, dim, l2_scale, l1_scale, drop,
                    batch_norm, custom, activation_fun = tf.nn.softplus,
                    kernel_initializer = None, bias_initializer = None, initializer = None, epsilon = 1e-8):
    """
    Input
        h: hidden layer
        var_dependency: True if varince is function of h
        var_type: diagonal, scalar, identity
        activation_fun: activation fun for the full connected layer predicting variance
        kernel_initializer: None if var_dependency is False
        bias_initializer: None if var_dependency is False
        initializer: None if var_dependency is True
    Return
        tensor
    """

    name = 'var'

    if var_dependency:

        if var_type == 'fullcov':

            h_var_diag,_,_ = layer(h, dim, fun = activation_fun, name = "{}_{}".format(name,'covar_vector'), l2_scale = l2_scale, l1_scale = l1_scale,
                      drop = drop,  batch_norm = batch_norm, custom = custom, kernel_initializer = kernel_initializer, bias_initializer = bias_initializer)

            h_var,_,_ = layer(h, dim * dim, fun = None, name = "{}_{}".format(name,'covar_vector'), l2_scale = l2_scale, l1_scale = l1_scale,
                      drop = drop,  batch_norm = batch_norm, custom = custom, kernel_initializer = kernel_initializer, bias_initializer = bias_initializer)

            h_var = tf.reshape(h_var, tf.shape(h)[:-1] + [dim, dim],
                               name = "{}_{}".format(name,'CovarianceMatrix'))

            h_var = tf.linalg.set_diag(h_var,h_var_diag,
                                       name = "{}_{}".format(name,'CovarianceMatrix'))

        elif var_type == 'identity':
                h_var = tf.ones(dim, name = 'var_diagonal')


        elif var_type == 'scalar':
            h_var_scalar,_,_ = layer(h, 1, fun = activation_fun, name = "{}_{}".format(name,'scalar'), l2_scale = l2_scale, l1_scale = l1_scale,
                      drop = drop,  batch_norm = batch_norm, custom = custom, kernel_initializer = kernel_initializer, bias_initializer = bias_initializer)
            h_var = tf.multiply(h_var_scalar, tf.ones(dim), name = 'diagonal')


        elif var_type == 'diagonal':
            h_var,_,_ = layer(h, dim, fun = activation_fun, name = "{}_{}".format(name,'diagonal'), l2_scale = l2_scale, l1_scale = l1_scale,
                      drop = drop,  batch_norm = batch_norm, custom = custom, kernel_initializer = kernel_initializer, bias_initializer = bias_initializer)


    else:

        # Set up variance first
        if var_type == 'identity':
                h_var = tf.ones(dim, name = 'var_diagonal')

        elif var_type == 'scalar':
                h_var_scalar = tf.get_variable("var_scalar", shape=[],
                                              initializer=initializer)
                h_var = tf.ones(dim) * h_var_scalar

        elif var_type == 'diagonal':
                h_var = tf.get_variable("var_diagonal", shape=[dim],
                                              initializer=initializer)

    return tf.add(h_var, epsilon)


class Distribution():

    """ Create probability layer """

    def __init__(self, h, dim, h_mu = None, h_std = None, fun = None, name = None, l2_scale = 1e-50, l1_scale = 1e-50,
                     drop = None,  batch_norm = None, custom = False,
                     epsilon = 1e-5, var_dependency = True, var_type = 'diagonal', activation_fun = tf.nn.softplus):

        if not var_type in ['scalar','diagonal','deterministic','identity']:
            raise Exception('Invalid input for var_type({})'.format(var_type))

        if h_mu is None:
            with tf.variable_scope('mu'):
                    h_mu,_,_ = layer(h, dim, fun = fun, name = "{}_{}".format(name,'mu'),
                                     l2_scale = l2_scale, l1_scale = l1_scale,
                                     drop = drop,  batch_norm = batch_norm, custom = custom)

        if var_type == 'deterministic':

            h_sample = h_mu
            h_std = None
            h_var = None
            h_logvar = None
            h_epsilon = None
            h_dist = None

        else:
            with tf.variable_scope('var'):

                if h_std is None:

                    h_std = create_variance(h = h, dim = dim, var_dependency = var_dependency, var_type = var_type, activation_fun = activation_fun,
                                            batch_norm = None, custom = custom, l2_scale = 1e-50, l1_scale = 1e-50, drop = None,
                                            kernel_initializer = tf.zeros_initializer,
                                            bias_initializer = tf.zeros_initializer,
                                            epsilon = epsilon)

                h_var = tf.square(h_std)
                h_logvar = tf.log(h_var + epsilon)

            if var_type == 'fullcov':
                h_dist = tfp.distributions.MultivariateNormalFullCovariance(loc = h_mu,
                                                                            covariance_matrix = h_std,
                                                                            allow_nan_stats = False,
                                                                            validate_args = True,
                                                                            name = 'distribution')
            else:
                h_dist = tfp.distributions.MultivariateNormalDiag(loc = h_mu,
                                                                  scale_diag = h_std,
                                                                  allow_nan_stats = False,
                                                                  validate_args = True,
                                                                  name = 'distribution')

            h_sample = h_dist.sample()
            h_epsilon = tf.zeros(tf.shape(h_mu))

        self.dist    = h_dist
        self.sample  = h_sample
        self.mu      = h_mu
        self.std     = h_std
        self.var     = h_var
        self.logvar  = h_logvar
        self.epsilon = h_epsilon

    def get_tensors(self):
        return [self.sample, self.mu, self.var, self.logvar, self.epsilon]
