import os
import time
import logging

import numpy as np

from siVAE import FeatureImportance as FI

def residual_analysis(self, y_reconstruct, y_test, min_PVE = 0, take_mean = False, verbose = False, save_PVE = False):
    """
    Calculate PVE and RE of the true and reconstructed numpy array
    Return [RE/PVE, pathway] if y_shape == 3 else [RE/PVE]
    """

    y_residual = y_test - y_reconstruct

    num_nn = y_residual.shape[0]

    output = []

    ## Reconstruction Error calculation
    RE = np.square(y_residual).mean(-2) # MSE per feature
    RE_mean = y_residual.mean(-2) # mean of residual per feature
    RE0 = np.square(y_test).mean(-2) # mean of test value per feature
    logging.info("RE: {}".format(RE))

    if take_mean:
        RE = RE.mean(-1)

    ## PVE calculation
    TV = np.square(y_test.std(axis = -2)) # [.. x sample x gene] -> [.. x gene]
    RV = np.square(y_residual.std(axis = -2)) # [.. x sample x gene] ->[.. x gene]

    if save_PVE:

        PVE = ((TV - RV) / TV)
        if verbose: logging.info("PVE: {}".format(PVE))
        if take_mean:
            PVE = PVE.mean(-1)

        output = (RE, PVE)
    else:
        if TV.shape[0] == 1:
            TV = np.tile(TV,(num_nn,1))
        if RE0.shape[0] == 1:
            RE0 = np.tile(RE0,(num_nn,1))

        output = (RE, RV, TV, RE0, RE_mean)

    return output


def run_FI(input, target, sample_DE, de, method_DE, feed_dict, kwargs_FI = {}, masks = None):
    """ """
    attributions_dict = {}
    start_time = time.time()

    attributions = FI.DE_marginal(input     = input,
                                  target    = target,
                                  sample    = sample_DE,
                                  de        = de,
                                  method_DE = method_DE,
                                  feed_dict = feed_dict,
                                  masks     = masks,
                                  **kwargs_FI)

    duration = time.time() - start_time

    attributions_dict['input'] = sample_DE
    attributions_dict['score'] = attributions
    attributions_dict['duration'] = duration

    return attributions_dict


def calculate_decoder_layers(model, sampled_outputs):
    """ Calculate the output of decoder layers """

    decoder_layers_list = []

    for ii in range(len(sampled_outputs)):

        sampled_output = sampled_outputs[ii,]

        feed_dict_full = {model.VAE_sample.X  : sampled_output,
                          model.VAE_sample.X_target : sampled_output,
                          model.VAE_feature.X : model.VAE_dict['feature'].data_handler.X}

        z_mu               = model.VAE_sample.calculate_z_mu(feed_dict_full)
        decoder_layers     = model.VAE_sample.calculate_decoder_layers(feed_dict_full)
        X_mu               = np.array(model.reconstruct_X(feed_dict_full)) # Don't sample X
        decoder_layers     = [z_mu] + decoder_layers
        decoder_layers[-1] = X_mu
        decoder_layers_list.append(decoder_layers)

    return decoder_layers
