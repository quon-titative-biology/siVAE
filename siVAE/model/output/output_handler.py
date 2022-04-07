import pickle
import gzip
import bz2

import pandas as pd
import numpy as np

from .degree_centrality import calculate_degree_centrality
from . import analysis

class output_handler():
    """
    Stores output of siVAE/VAE results
    Inputs:
        result_dict:
    """

    def __init__(self, result_dict = None, save_dir = None, model_name = 'siVAE'):
        if result_dict is not None:
            self.result_dict = result_dict
        elif save_dir is not None:
            self.load_result(save_dir,)

        self.model_name = "siVAE"

    def __getitem__(self,i):
        return self.result_dict[i]

    def add_result(self,key,val):
        """ Add key/val to the result_dictionary """
        self.result_dict[key] = val

    def update_result(self,dict_new):
        """ Update the result dictionary with an input dictionary """
        self.result_dict.update(dict_new)

    def get_dict(self):
        return self.result_dict

    def keys(self):
        return self.get_dict().keys()

    def get_value(self,key,get_model=True):
        return self.get_model().result_dict[key]

    def save(self,filename,**kwargs):
        save_pickle(self,filename,**kwargs)

    def load(self,filename,**kwargs):
        self.result_dict = load_pickle(filename,**kwargs)

    def convert_scalars_to_df(self):
        if 'scalars' in self.result_dict.keys():
            self.add_result('scalars_df',scalars2df(self.get_value('scalars')))

    def get_model(self):
        if 'model' in self.result_dict.keys():
            return self.get_dict()['model']
        else:
            return self

    def calculate_degree_centrality(self, method='ReconError', in_place=False):

        if method == 'ReconError':
            ReconError = np.square(self.get_value('reconstruction')[0] - self.get_value('reconstruction')[1]).transpose() # Transpose to n_feature X n_sample
            degree_centrality = calculate_degree_centrality(recon_loss=ReconError)
        elif method == 'DistanceFromCenter':
            embeddings = self.get_value('latent_embedding')['feature']
            degree_centrality = calculate_degree_centrality(embeddings = embeddings)
        else:
            raise Exception('Input Valid Method for Degree Centrality Calculation')

        if in_place:
            self.add_result('degree_centrality',degree_centrality)
        else:
            return degree_centrality

    def create_values_dict(self):
        return extract_value(self.get_model().get_dict())

    def get_feature_embeddings(self):
        return self.get_model().get_value('latent_embedding')['feature']

    def get_sample_embeddings(self):
        return self.get_model().get_value('latent_embedding')['sample']

    def get_feature_attributions(self):
        """"""
        return get_feature_attributions(self)

    def save_losses(self,logdir):
        analysis.save_losses(self,logdir)


def get_feature_attributions(siVAE_result):
    """
    Return feature attribution dictionary with keys encoder/decoder that
    maps to n_methods x n_samples x LE_dim x n_features array.
    """

    self = siVAE_result
    if 'sample_dict' in self.get_model().get_dict().keys():

        sample_dict = self.get_model().get_value('sample_dict')

        if 'attributions_samples' in sample_dict.keys():

            attrb_dict = sample_dict['attributions_samples']
            feature_attribution = {}

            for AE_part,FA_dict in attrb_dict.items():
                FA_scores  = FA_dict['score']

                if AE_part == 'encoder':
                    FA_scores = np.swapaxes(FA_scores,2,3)

                FA_methods = FA_dict['methods']
                feature_attribution[AE_part] = FA_scores

    else:

        raise Exception('sample_dict not present. Make sure the model has been run with do_FA=True')

    return feature_attribution, FA_methods


def save_pickle(obj, filename, zip_method = None, protocol=-1, correct_suffix = True):
    """ """
    if correct_suffix:
        filename = filename.rsplit(".",1)[0]
        if zip_method is None:
            filename += ".pickle"
        elif zip_method is 'gzip':
            filename += ".pgz"
        elif zip_method is 'bz2':
            filename += ".pbz2"
        else:
            raise Exception('Specify a correct zip_method: {}'.format(zip_method))

    if zip_method is None:
        with open(filename, 'wb') as f:
            pickle.dump(obj, f, protocol)
    elif zip_method == 'gzip':
        with gzip.open(filename, 'w') as f:
            pickle.dump(obj, f)
    elif zip_method == 'bz2':
         with bz2.BZ2File(filename, 'w') as f:
            pickle.dump(obj, f)
    else:
        raise Exception('Specify a correct zip_method: {}'.format(zip_method))


def load_pickle(filename, zip_method=None, correct_suffix = True):

    if correct_suffix:
        filename = filename.rsplit(".",1)[0]
        if zip_method is None:
            filename += ".pickle"
        elif zip_method is 'gzip':
            filename += ".pgz"
        elif zip_method is 'bz2':
            filename += ".pbz2"
        else:
            raise Exception('Specify a correct zip_method: {}'.format(zip_method))

    if zip_method is None:
        with open(filename, 'rb') as f:
            loaded = pickle.load(f)
    elif zip_method == 'gzip':
        with gzip.open(filename, 'r') as f:
            loaded = pickle.load(f)
    elif zip_method == 'bz2':
         with bz2.BZ2File(filename, 'r') as f:
            loaded = pickle.load(f)
    else:
        raise Exception('Specify a correct zip_method: {}'.format(zip_method))

    return loaded


def tbscalars2df(scalars_dict):
    """ Convert dictionary from tensorboard scalars to data frame """
    df_tb = pd.concat([pd.DataFrame({'Step': arrays[0], 'Value': arrays[1], 'Tag':tag}) for tag,arrays in scalars_dict.items()])
    df_tb['Model'] = [tag.split('/')[0] for tag in df_tb.Tag]
    df_tb['Name'] = [tag.split('/')[-1] for tag in df_tb.Tag]
    return df_tb


def scalars2df(scalars_dict):
    """"""
    df_tb_list = []
    for key,scalars_d in scalars_dict.items():
        df_tb = tbscalars2df(scalars_d)
        df_tb.insert(0,'Type',key)
        df_tb_list.append(df_tb)
    return pd.concat(df_tb_list)


def extract_value(result_dict):
    """ """
    # result_dict = output_handler.get_dict()
    # result  = output_handler.get_model().get_dict()
    result  = result_dict
    if 'sample' in result_dict.keys():
        z_mu    = result['z_mu']
        z_var   = result['y_mu']
        X       = result['reconstruction'][0]
        X_recon = result['reconstruction'][1]
        values_dict = {'model': {'z_mu'   : z_mu,
                                 'z_var'  : z_var,
                                 'X'      : X,
                                 'X_recon': X_recon}}
    else:
        z_mu    = result['latent_embedding']['sample']
        z_var   = result['latent_embedding_var']['sample']
        v_mu    = result['latent_embedding']['feature']
        v_var   = result['latent_embedding_var']['feature']
        X       = result['reconstruction'][0]
        X_recon = result['reconstruction'][1]
        #
        values_dict = {'model': {'z_mu'   : z_mu,
                                 'z_var'  : z_var,
                                 'v_mu'   : v_mu,
                                 'v_var'  : v_var,
                                 'X'      : X,
                                 'X_recon': X_recon}}

        # Decoder layers in sample
        if 'decoder_layers' in result.keys():
            decoder_layers_sample = result['decoder_layers']['sample'][:-1]
            decoder_layers_feature = result['decoder_layers']['feature']
            decoder_layers_sample  = [z_mu] + decoder_layers_sample
            decoder_layers_feature = [v_mu] + decoder_layers_feature
            values_dict['model']['decoder_layers'] = {'sample' :decoder_layers_sample,
                                                      'feature':decoder_layers_feature}

    # Feature Attributions
    if 'sample_dict' in result.keys():
        sample_dict = result['sample_dict']
        if 'attributions_samples' in sample_dict.keys():
            attrb_dict = sample_dict['attributions_samples']
            values_dict['model']['Feature Attribution'] = {}
            for AE_part,FA_dict in attrb_dict.items():
                FA_scores  = FA_dict['score']
                if AE_part == 'encoder':
                    FA_scores = np.swapaxes(FA_scores,2,3)
                FA_methods = FA_dict['methods']
                values_dict['model']['Feature Attribution'][AE_part] = FA_scores
            values_dict['model']['Feature Attribution Methods'] = FA_methods

    # Sample-wise VAE
    if 'sample' in result_dict.keys():
        result_sample = result_dict['sample']
        if result_sample is not None:
            z_mu  = result_sample['z_mu']
            z_var = result_sample['z_var']
            values_dict['sample'] = {'z_mu' : z_mu,
                                     'z_var': z_var}
    # Feature-wise VAE
    if 'feature' in result_dict.keys():
        result_feature = result_dict['feature']
        if result_feature is not None:
            z_mu  = result_feature['z_mu']
            z_var = result_feature['z_var']
            values_dict['feature'] = {'z_mu' : z_mu,
                                      'z_var': z_var}

    return values_dict
