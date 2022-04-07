####

import os
import numpy as np

from siVAE import util
from siVAE.data.data_handler import convert2raw

## Set directory for saving siVAE specific inputs to GRN/gene relevance
save_dir = 'out/data/siVAE'
os.makedirs(save_dir,exist_ok=True)

## =============================================================================
## Save expression matrix
## =============================================================================

## Save the train/test split for consistency
datadir = os.path.join('out/data_dict.pickle')
if os.path.exists(datadir):
    data_dict = util.load_pickle(datadir)

adata     = data_dict['sample'].X
adata_raw = convert2raw(adata)

gene_names = adata_raw.var_names.to_numpy()
cell_names = adata_raw.obs['Labels'].to_numpy()

np.savez(file       = 'out/data/expression.npz',
         exprs      = adata_raw.X,
         cell_names = cell_names,
         gene_names = gene_names)


## =============================================================================
## Load siVAE results
## =============================================================================

siVAE_result_dir = 'out/siVAE/kfold-0/siVAE_result.pickle'
siVAE_result     = util.load_pickle(siVAE_result_dir)

## =============================================================================
## Save latent embedding for gene relevance input
## =============================================================================

gene_relevance_input_dir = os.path.join(save_dir,'gene_relevance_input.npz')

cell_embedding = siVAE_result.get_model().get_sample_embeddings()

np.savez(file       = gene_relevance_input_dir,
         exprs      = adata_raw.X,
         cell_names = cell_names,
         gene_names = gene_names,
         coords     = cell_embedding)


## =============================================================================
## Save hidden layers and reconstructed expression of siVAE
## =============================================================================

## Load hidden_layers

hidden_layers  = siVAE_result.get_model().get_value('decoder_layers')['feature']
gene_embedding = siVAE_result.get_model().get_feature_embeddings()

hidden_layers.insert(0,gene_embedding)

## Save hidden_layers
for ii,hl in enumerate(hidden_layers):
    hl = hl.transpose()
    filename = 'siVAE-l{}'.format(ii)
    np.savez(file       = os.path.join(save_dir, filename),
             exprs      = hl,
             cell_names = np.arange(hl.shape[0]),
             gene_names = gene_names)

## Load and save reconstruction
reconstruction = siVAE_result.get_model().get_value('reconstruction')[1]

np.savez(file       = os.path.join(save_dir,'reconstructed.npz'),
         exprs      = reconstruction,
         cell_names = cell_names,
         gene_names = gene_names)
