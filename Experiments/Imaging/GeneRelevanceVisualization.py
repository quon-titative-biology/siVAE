import os

import numpy as np

from siVAE.model.output.analysis import infer_FA_loadings
from siVAE.model.plot import plot_images
from siVAE.model.output import analysis

# Plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns


## Load directories
logdir_GR = 'out/siVAE/gene_relevance'
logdir_list = os.listdir(logdir_GR)
logdir_list.sort()

logdir='out/siVAE'

## Set parameters
infer_FA_method=1

##
all_loadings = None
methods_loadings = []

for k in logdir_list:
    #---#
    logdir_K = os.path.join(logdir_GR,k)
    #---#
    ## Load gene relevance results and convert to loadings
    npz = np.load(os.path.join(logdir_K,'gene_relevance_result.npz'),allow_pickle=True)
    #---#
    gene_names = npz['genes']
    gene_relevance = np.expand_dims(npz['partials'],0)
    gene_relevance = np.moveaxis(gene_relevance,3,1)
    gene_relevance = np.nan_to_num(gene_relevance,0)
    gr_loadings = analysis.infer_FA_loadings(gene_relevance,
                                             method=infer_FA_method)
    #---#
    ## Concatenate the loadings
    if all_loadings is None:
        all_loadings = gr_loadings
    else:
        all_loadings = np.concatenate([all_loadings,gr_loadings],axis=0)
    methods_loadings += ['Gene Relevance ({})'.format(k)]
    #---#

ImageDims=[28,28,1]

images = all_loadings
images_list = images.reshape(*images.shape[:-1],*ImageDims)
# rownames = methods_loadings
# colnames = ['Methods'] + ['dim-'+str(dim+1) for dim in range(len(images_list))]
colnames = ['Methods'] + methods_loadings
rownames = ['dim-'+str(dim+1) for dim in range(len(images_list[0]))]
_ = plot_images(images_list,colnames=colnames,rownames=rownames,cutoff=0.01)
figname = os.path.join(logdir,"GeneRelevance_visualization.pdf")
plt.savefig(figname)
plt.close()
