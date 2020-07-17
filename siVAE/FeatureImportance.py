import os
import time
import logging

from deepexplain.tensorflow import DeepExplain

import numpy as np

dict_DE = {'Saliency Maps'       : 'saliency',
           'Gradient * Input'    : 'grad*input',
           'Integrated Gradients': 'intgrad',
           'Epsilon-LRP'         : 'elrp',
           'DeepLIFT'            : 'deeplift',
           'Occlusion'           : 'occlusion',
           'Shapley Value'       : 'shapley_sampling'}

def DE(input, target, sample, de, mask = None, method_DE = ['Saliency Maps', 'Gradient * Input', 'Integrated Gradients', 'DeepLIFT', 'Epsilon-LRP', 'Occlusion'], feed_dict = {}, **kwargs):
    """"""
    if len(target.shape) == 0:
        mask = 1
    else:
        if mask is None:
            mask = np.ones(target.shape[1])

    attributions = {}

    for method in method_DE:
        print(method)
        print(dict_DE[method])
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

        attributions[method] = de.explain(method = dict_DE[method],
                                          T = target*mask,
                                          X = input,
                                          xs = sample,
                                          feed_dict = feed_dict,
                                          **kwargs_in)
    return attributions


def DE_marginal(input, target, sample, de, masks = None, method_DE = ['Saliency Maps', 'Gradient * Input', 'Integrated Gradients', 'DeepLIFT', 'Epsilon-LRP', 'Occlusion'], feed_dict = {}, **kwargs):
    """"""
    marginal = []

    if len(target.shape) in [0,1]:
        raise Exception('Shape of target must be [num_sample,num_features]')
    else:
        n_node = int(target.shape[1])

    if masks is None:
        masks = [np.arange(n_node) == ii for ii in range(n_node)]
    elif masks == 'sum':
        masks = [np.ones(n_node)]
    # else:
    #     raise Exception('Invalid input for masks')

    for mask in masks:
        marginal.append(DE(input     = input,
                           target    = target,
                           sample    = sample,
                           de        = de,
                           mask      = mask,
                           method_DE = method_DE,
                           feed_dict = feed_dict,
                           **kwargs))

    return(marginal)



    ## Custom: limit operations to those downstream of the input
    print('Custom: limit operations to those downstream of the input')
    ops_check = [descendants(op) for op in self.X.consumers()]
    ops_check = list(set(chain.from_iterable(ops_check)))


def transform_dict(attributions_dict):
    """
    returns [method x sample x input x target]
    """
    scores = attributions_dict['score']
    method_DE = [key for key in scores[0].keys()]
    scores_array = np.array([[score[method] for score in scores] for method in method_DE])
    scores_array = np.transpose(scores_array,[0,2,3,1])
    return method_DE, scores_array
