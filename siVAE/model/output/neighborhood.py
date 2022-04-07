"""
Create
"""

import os,gc

import numpy as np
import pandas as pd

import tensorflow as tf

from siVAE.data import data_handler as dh
from siVAE.model import VAE

def calculate_prediction_accuracy(similarity_matrix, data_handler, model_kwargs={}, split_index_list=None, num_neighbors=20, logdir=""):
    """
    Input:
        similarity_matrix: n_query x n_target data frame with index and columns
        data_handler:
        model_kwargs: dictionary specifying parameters for the prediction model
        num_neighbors: integer indicating number of neighbors
        split_index_list: specify the train/test split
    """

    ## Create data handler object with pre-allocation
    datah = dh.data_handler(X=data_handler.X[:,:num_neighbors],y=data_handler.X[:,[1]])
    ## Set up architecture of hidden layers
    LE_dim       = model_kwargs['LE_dim']
    architecture = model_kwargs['architecture']

    # For datah_sample
    h_dims = architecture.replace('LE',str(LE_dim))
    h_dims = [int(dim) for dim in h_dims.split("-")]
    datah.index_latent_embedding = int(h_dims.pop())
    datah.h_dims = h_dims

    if split_index_list is None:
        datah.create_split_index_list(k_split=0.8,random_seed=0)
    else:
        datah.split_index_list = split_index_list

    ## Iterate through the query genes
    scores_train = []
    scores_test  = []
    for g,v_similarity in similarity_matrix.iterrows():

        ## Subset the matrix based on similarity matrix by selecting the neighbors with highest values
        v_similarity = v_similarity[v_similarity.index != g]
        genes_v = v_similarity.sort_values(ascending=False)[:num_neighbors].index.to_numpy()
        gnames = data_handler.X.var_names

        datah.X = data_handler.X[:,np.isin(gnames,genes_v)]
        datah.y = data_handler.X[:,np.isin(gnames,g)]
        datah.create_dataset(kfold_idx=0)

        ## Set logdir for tensorboard/tensorflow
        model_kwargs['logdir_tf'] = os.path.join(logdir,
                                                 'Gene-{}'.format(g))
        os.makedirs(model_kwargs['logdir_tf'],exist_ok=True)

        ## Run model on tensorflow
        tf.compat.v1.reset_default_graph()
        tf.reset_default_graph()
        with tf.Session() as sess:
            model = VAE.AutoEncoder(data_handler = datah, **model_kwargs)
            model.build_model(reset_graph = False)
            result = model.train(sess,initialize=True)
            sess.close()

        ## Record the accuracy
        score_train = result.get_value('losses')['train'][2]
        score_test  = result.get_value('losses')['test'][2]
        scores_train.append(score_train)
        scores_test.append(score_test)

        gc.collect()

    df_score = pd.DataFrame({'Gene' : similarity_matrix.index,
                             'Train': scores_train,
                             'Test' : scores_test},
                             index = similarity_matrix.index)

    return df_score


def extract_similarity_matrix(result_dict, do_siVAE=True, do_FA=False, do_layers=False,
                              method_DE=None, method='',category='', grn_mat_dict={}, method2category={}):
    """
    do_FA: extract loadings inferred from feature attribution
    """

    values_dict_ = result_dict.create_values_dict()
    gene_names = result_dict.get_model().get_value('var_names')

    if category == '':
        category=method
    method_name = method

    ## Extract siVAE Loadings
    if do_siVAE:
        v_mu = values_dict_['model']['v_mu'].transpose()
        df_mat = -loadings2dist(v_mu.transpose(),gene_names)
        grn_mat_dict[method_name] = df_mat
        method2category[method_name] = category

    ## Extract Loadings from Feature Attribution
    if do_FA:
        if method_DE is None:
            method_DE = ['SaliencyMaps','GradInput','DeepLIFT']
        scores = values_dict_['model']['Feature Attribution']['decoder']
        infer_FA_method=1
        FA_loadings = analysis.infer_FA_loadings(np.swapaxes(scores,1,2),
                                                       method=infer_FA_method)
        ## Feature attributions loadings
        FA_loadings_ = np.swapaxes(FA_loadings,1,2)
        FA_loadings = FA_loadings[::-1]
        method_DE = method_DE[::-1]
        method_FA = ["{}-{}".format(method,m) for m in method_DE]
        for FA_method,loadings in zip(method_FA,FA_loadings_):
            df_mat = -loadings2dist(loadings,gene_names,gene_names)
            grn_mat_dict[FA_method] = df_mat
            method2category[FA_method] = category

    if do_layers:
        for ii, loadings in enumerate(result_dict.get_model().get_value('decoder_layers')['feature']):
            layer_name="{}-L{}(n={})".format(method,ii+2,loadings.shape[-1])
            df_mat = -loadings2dist(loadings,gene_names,gene_names)
            grn_mat_dict[layer_name] = df_mat
            method2category[layer_name] = category

    return grn_mat_dict, method2category


def loadings2dist(loadings,gene_names,gene_names2=None):
    """
    Input:
        gene embeddings/loadings
    Return:
        n_gene X n_gene matrix showing distance between genes
    """

    df_mat = pd.DataFrame([np.linalg.norm(loadings-g,axis=1) for g in loadings],
                          index = gene_names,
                          columns = gene_names)

    if gene_names2 is not None:
        df_mat = df_mat.reindex(gene_names2).transpose().reindex(gene_names2).transpose().fillna(0)

    return df_mat


def identify_neighborhood_genes(similarity_matrix):
    pass
