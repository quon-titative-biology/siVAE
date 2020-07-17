import os
import time
import logging

import tensorflow as tf

import numpy as np


def print_losses(losses, losses_name):
    """ Print losses in a specific format """
    zipped = zip(losses_name, losses)
    format_args = [item for pack in zipped for item in pack]
    str_report = "Train: {}={:.4}" + ", {}={:.4}" * (len(losses_name)-1)
    logging.info(str_report.format(*format_args))


def scope_name(scope):
    """ Returns scope name """

    scope_valid = False
    count = 0
    root = scope

    while not scope_valid:
        theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                  scope = os.path.join(tf.get_variable_scope().name,scope))
        if len(theta) == 0:
            scope_valid = True
        else:
            count += 1
            scope = '{}_{}'.format(root, count)

    return scope


def reshape_sum(X, ImageDims):
    X_shape = X.shape[:-1]
    X_image = X.reshape(*X_shape,*ImageDims)
    X_image = X_image.sum(-1)
    X_image = X_image.reshape(*X_shape,*ImageDims[:-1],1)
    return X_image


class batch_handler():
    """ Handles mini-batches during model training """

    def __init__(self, total_size, mb_size):
        self.total_size = total_size
        self.idx_all = np.random.permutation(np.array(range(total_size)))
        self.i_epoch = 0
        self.mb_size = mb_size

    def next_batch(self):
        # input: idx_all, num, i_epoch
        # output: idx, ranom selection of the data
        # use in loop while updating idx_all and i_epoch
        start = self.i_epoch
        num = self.mb_size
        total_size = self.total_size
        idx_all = self.idx_all

        end = start + num
        new_i_epoch = end % total_size

        if end > total_size:
            idx_before_shuffle = idx_all[start:]
            idx_all = np.random.permutation(idx_all)
            end = new_i_epoch
            idx_after_shuffle = idx_all[:end]
            idx_out = np.concatenate((idx_before_shuffle, idx_after_shuffle))
        else:
            idx_out = idx_all[start:end]

        self.idx_all = idx_all
        self.i_epoch = new_i_epoch

        return idx_out
