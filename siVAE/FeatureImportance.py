import os
import time
import logging

from deepexplain.tensorflow import DeepExplain
from tensorflow_forward_ad import forward_gradients

import numpy as np

import tensorflow as tf

def DE(input, target, sample, method, de, mask = None,  feed_dict = {}, **kwargs):
    """

    Parameters
    ----------
    input
        Input tensor
    target
        Target tensor
    sample
        Sample numpy array
    de
        deepexplain object
    mask
        mask the length of target tensor
    method_DE
        List of methods for feature attribution
        ['Saliency Maps', 'Gradient * Input', 'Integrated Gradients', 'DeepLIFT', 'Epsilon-LRP', 'Occlusion']
    feed_dict
        Additional feed_dict arguments

    Return
    ------
    Dictionary that maps method to feature attribution that is [sample x input]
    """

    dict_DE = {'Saliency Maps'       : 'saliency',
               'Gradient * Input'    : 'grad*input',
               'Integrated Gradients': 'intgrad',
               'Epsilon-LRP'         : 'elrp',
               'DeepLIFT'            : 'deeplift',
               'Occlusion'           : 'occlusion',
               'Shapley Value'       : 'shapley_sampling'}

    if len(target.shape) == 0:
        mask = 1
    else:
        if mask is None:
            mask = np.ones(target.shape[1])

    attributions = {}

    kwargs_in = {}
    if method in ['DeepLIFT','Integrated Gradients']:
        if 'baseline' in kwargs.keys():
            kwargs_in['baseline'] = kwargs['baseline']
    elif method in ['Integrated Gradients']:
        if 'steps' in kwargs.keys():
            kwargs_in['steps'] = kwargs['steps']
    elif method in ['DeepLIFT']:
        if 'init_ref' in kwargs.keys():
            kwargs_in['init_ref'] = kwargs['init_ref']

    attributions = de.explain(method = dict_DE[method],
                              T = target * mask,
                              X = input,
                              xs = sample,
                              feed_dict = feed_dict,
                              **kwargs_in)

    return attributions


def DE_marginal(input, target, sample, method, de, masks = None, feed_dict = {}, **kwargs):
    """
    Parameters

    ----------
    input
        Input tensor
    target
        Target tensor
    sample
        Sample numpy array
    method
        method for feature attribution
        ['Saliency Maps', 'Gradient * Input', 'Integrated Gradients', 'DeepLIFT', 'Epsilon-LRP', 'Occlusion']
    de
        deepexplain object
    masks
        List of mask the length of target tensor
    feed_dict
        Additional feed_dict arguments
    """

    dict_DE = {'Saliency Maps'       : 'saliency',
               'Gradient * Input'    : 'grad*input',
               'Integrated Gradients': 'intgrad',
               'Epsilon-LRP'         : 'elrp',
               'DeepLIFT'            : 'deeplift',
               'Occlusion'           : 'occlusion',
               'Shapley Value'       : 'shapley_sampling'}

    marginal = []

    if len(target.shape) in [0,1]:
        raise Exception('Shape of target must be [num_sample,num_features]')
    else:
        n_node = int(target.shape[1])

    if masks is None:
        masks = [np.arange(n_node) == ii for ii in range(n_node)]
    elif masks == 'sum':
        masks = [np.ones(n_node)]

    kwargs_in = {}
    if method in ['DeepLIFT','Integrated Gradients']:
        if 'baseline' in kwargs.keys():
            kwargs_in['baseline'] = kwargs['baseline']
    elif method in ['Integrated Gradients']:
        if 'steps' in kwargs.keys():
            kwargs_in['steps'] = kwargs['steps']
    elif method in ['DeepLIFT']:
        if 'init_ref' in kwargs.keys():
            kwargs_in['init_ref'] = kwargs['init_ref']

    explainer = de.get_explainer(method = dict_DE[method],
                                 T = target,
                                 X = input,
                                 feed_dict = feed_dict,
                                 **kwargs_in)
    import time

    for ii,mask in enumerate(masks):
        start_time = time.time()
        attributions = explainer.run(xs = sample,
                                     ys = [mask] * len(sample))
        marginal.append(attributions)
        dur = time.time() - start_time
        with open('file.txt', 'a') as file:
            file.write("Iter {}: {} s".format(ii,dur) +'\n')
        # marginal.append(DE(input     = input,
        #                    target    = target,
        #                    sample    = sample,
        #                    de        = de,
        #                    mask      = mask,
        #                    method    = method,
        #                    feed_dict = feed_dict,
        #                    **kwargs))

    marginal = np.array(marginal)        # [n_target, n_sample, n_input]
    marginal = np.moveaxis(marginal,0,2) # [n_sample, n_input, n_target]

    return(marginal)


def FG(input, target, sample, method, sess, masks = None, feed_dict = {}, **kwargs):
    """
    Parameters
    ----------
    input
        Input tensor
    target
        Target tensor
    sample
        Sample numpy array
    de
        deepexplain object
    masks
        List of mask the length of target tensor
    method_DE
        List of methods for feature attribution
        ['Saliency Maps', 'Gradient * Input']
    feed_dict
        Additional feed_dict arguments
    """

    if len(input.shape) in [0,1]:
        raise Exception('Shape of target must be [num_sample,num_features]')
    else:
        n_node = int(input.shape[1])

    logging.info('n_node={}'.format(n_node))

    ## Set up masks
    if masks is None:
        masks = [np.arange(n_node) == ii for ii in range(n_node)]
    elif masks == 'sum':
        masks = [np.ones(n_node)]

    ## Run forward gradients on tensorflow
    #### 1
    feed_dict_in = feed_dict.copy()
    feed_dict_in[input] = sample

    marginal = []

    for mask in masks:

        v = tf.convert_to_tensor(mask.reshape(1,-1), dtype=tf.float32)
        dydx = forward_gradients(target,input,v)[0]

        if method == 'Saliency Maps':
            FA = dydx
        elif method == 'Gradient * Input':
            FA = dydx * target
        else:
            raise Exception('Invalid feature attribution method')

        attributions = sess.run(FA, feed_dict = feed_dict_in)

        marginal.append(attributions)

    #### 2
    # v = tf.placeholder(tf.float32,[1,int(input.shape[1])])
    #
    # dydx = forward_gradients(target,input,v)[0]
    #
    # if method == 'Saliency Maps':
    #     FA = dydx
    # elif method == 'Gradient * Input':
    #     FA = dydx * target
    # else:
    #     raise Exception('Invalid feature attribution method')
    #
    # marginal = []
    #
    # for mask in masks:
    #     feed_dict_in = feed_dict.copy()
    #     feed_dict_in[input] = sample
    #     feed_dict_in[v]     = mask.reshape(1,-1)
    #     attributions = sess.run(FA, feed_dict = feed_dict_in)
    #
    #     marginal.append(attributions)

    ## Combine into one array
    marginal = np.array(marginal)        # [n_input, n_sample, n_target]
    marginal = np.moveaxis(marginal,0,1) # [n_sample, n_input, n_target]

    return(marginal)


def feature_attribution(input, target, sample, sess = None, de = None, masks = None, method_DE = ['Saliency Maps'], feed_dict = {}, mode = 'forward', **kwargs):
    """
    Parameters
    ----------
    input
        Input tensor
    target
        Target tensor
    sample
        Sample numpy array
    de
        deepexplain object
    masks
        List of mask the length of target tensor
    method_DE
        List of methods for feature attribution
        ['Saliency Maps', 'Gradient * Input', 'Integrated Gradients', 'DeepLIFT', 'Epsilon-LRP', 'Occlusion']
    feed_dict
        Additional feed_dict arguments

    Return
    ------
    attributions
        [ n_method x n_sample x n_input x n_target ]
    """

    forward_mode_methods = ['Saliency Maps', 'Gradient * Input']

    attributions = []

    for method in method_DE:

        if mode == 'forward' and method in forward_mode_methods:
            attribution = FG(input, target, sample, method, sess, masks = masks, feed_dict = feed_dict, **kwargs)
        else:
            attribution = DE_marginal(input, target, sample, method, de, masks = masks, feed_dict = feed_dict, **kwargs)

        attributions.append(attribution)

    attributions = np.array(attributions)

    return(attributions)
