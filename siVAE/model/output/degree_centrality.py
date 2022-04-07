import os

import numpy as np
import pandas as pd

def calculate_degree_centrality(embeddings=None,recon_loss=None,logdir_FA=None):
    ## 3 options ()
    # 1. Distance from origin in gene embedding
    if embeddings is not None:
        siVAE_gene_score = -np.sqrt(np.square(embeddings).sum(1))
    # 2. Reconstruction loss per gene
    elif recon_loss is not None:
        siVAE_gene_score = 1-recon_loss
    # 3. Using feature attribution as GRN
    elif logdir_FA is not None:
        pass
        # result_dict_dir = os.path.join(logdir_FA,"result_dict.pickle")
        # # result_dict_FA = util.load_pickle(result_dict_dir)
        # attr = result_dict_FA['model']['sample_dict']['attributions_samples']
        # encoder_FA = attr['encoder']['score'][0]
        # decoder_FA = attr['decoder']['score'][0]
        # FA = 0
        # for e,d in zip(encoder_FA,decoder_FA):
        #     FA += np.abs(np.matmul(e,d))
        # FA = FA/encoder_FA.shape[0]
        # siVAE_gene_score = FA.sum(-1)
    degree_centrality = siVAE_gene_score
    return degree_centrality
