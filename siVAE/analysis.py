import os
import time
import logging

import itertools

# Math
import pandas as pd
import numpy as np

# Plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# Custom
from siVAE import util
from siVAE.util import reduce_dimensions
from siVAE.model.plot import plot_images

import scanpy as sc
import gseapy as gp

def getPCALoadings(pca):
    return((pca.components_.T * np.sqrt(pca.explained_variance_)).T)


def extract_value(result_dict):
    """ """
    result  = result_dict['model']
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
                FA_scores,FA_methods = reformat_FA(FA_dict['score'],AE_part)
                values_dict['model']['Feature Attribution'][AE_part] = FA_scores
            values_dict['model']['Feature Attribution Methods'] = FA_methods

    # Sample-wise VAE
    if 'sample' in result_dict.keys():
        result_sample = result_dict['sample']
        z_mu  = result_sample['z_mu']
        z_var = result_sample['z_var']
        values_dict['sample'] = {'z_mu' : z_mu,
                                 'z_var': z_var}
    # Feature-wise VAE
    if 'feature' in result_dict.keys():
        result_feature = result_dict['feature']
        z_mu  = result_feature['z_mu']
        z_var = result_feature['z_var']
        values_dict['feature'] = {'z_mu' : z_mu,
                                  'z_var': z_var}

    # Decoder layers
    result_dict['model']
    return values_dict


def reformat_FA(scores_list,AE_part):
    """
    Reformat output of feature attribution scores into
    [FI_method, LE, sample, feature]
    """
    FA_method_list = [m for m in scores_list[0].keys()]
    scores_array = np.array([[score for score in score_dict.values()] for score_dict in scores_list])
    scores_array = np.transpose(scores_array, axes = [2,1,0,3]) # Sample x FI_method x Features x LE
    if AE_part == 'decoder':
        scores_array = np.swapaxes(scores_array,2,3)
    scores_array_FA = np.transpose(scores_array,[1,2,0,3]) # FI_method x LE x sample x Features
    return scores_array_FA,FA_method_list


def infer_FA_loadings(scores):
    """ infer FA loadings from FA scores """
    loadings = []
    for scores_FA in scores:
        loadings_LE = []
        for scores_LE in scores_FA:
            pca = PCA(n_components = 1)
            FA_loadings = pca.fit_transform(scores_LE.T).T #
            loadings_LE.append(FA_loadings)
        loadings.append(loadings_LE)
    return(np.array(loadings).sum(axis=-2))


def plot_FA_loadings(values_dict,logdir,ImageDims):
    """ Plot the inferred FA loadings """
    FA_methods = values_dict['model']['Feature Attribution Methods']
    for AE_part,scores in values_dict['model']['Feature Attribution'].items():
        FA_loadings = infer_FA_loadings(scores)
        images_list = FA_loadings.reshape(*FA_loadings.shape[:-1],*ImageDims)
        colnames = ['Dimensions'] + FA_methods
        rownames = ['dim-'+str(dim+1) for dim in range(len(images_list[0]))]
        _ = plot_images(images_list,colnames=colnames,rownames=rownames,cutoff=0.1)
        figname = os.path.join(logdir,"Loadings-{}.pdf".format(AE_part))
        plt.savefig(figname)
        plt.close()


def plot_siVAE_loadings(values_dict, logdir, ImageDims):
    loadings = values_dict['model']['v_mu'].T
    images_list = [loadings.reshape(*loadings.shape[:-1],*ImageDims)]
    colnames = ['Dimensions','siVAE']
    rownames = ['dim-'+str(dim+1) for dim in range(len(images_list[0]))]
    _ = plot_images(images_list,colnames=colnames,rownames=rownames,cutoff=0.1)
    figname = os.path.join(logdir,"Loadings-siVAE.pdf")
    plt.savefig(figname)
    plt.close()


def loading_analysis(values_dict, plot_args_dict_sf, logdir, dims = 2, num_genes = 20, genesets = None, **kwargs):
    """  """

    labels_gene = plot_args_dict_sf['feature']['names']
    loadings = values_dict['model']['v_mu']

    for d in range(dims):
        loading = loadings[:,d]
        ranking = np.argsort(loading)[::-1]
        ranking = np.concatenate([ranking[:num_genes],ranking[-num_genes:]])
        loading_ranked = loading[ranking]
        labels_ranked = labels_gene[ranking]

        df = pd.DataFrame({'Genes':labels_ranked,'Contribution':loading_ranked})
        g = sns.barplot(x='Genes',y='Contribution',data=df,**kwargs)
        g.set_xticklabels(labels_ranked,rotation=90,size=5)
        figname = os.path.join(logdir,"Loadings_Feature-dim_{}.pdf".format(d+1))
        plt.tight_layout()
        plt.savefig(figname)
        plt.close()

        ## Save to CSV
        ranking = np.argsort(loading)[::-1]
        loading_ranked = loading[ranking]
        labels_ranked = labels_gene[ranking]
        df = pd.DataFrame({'Genes':labels_ranked,'Contribution':loading_ranked})
        df.to_csv(os.path.join(logdir,'Loadings_Feature-dim_{}.csv'.format(d+1)),sep=',',index=False,header=True)

        ## Perform GSEA
        if genesets is not None:
            for gs_name,geneset in genesets.items():
                outdir = os.path.join(logdir,'GSEA','siVAE',gs_name,'dim-{}'.format(d))
                pre_res = gp.prerank(rnk = df, gene_sets = geneset,
                                     outdir = outdir, max_size = 200)


    X = values_dict['model']['X']
    pca = PCA(dims)
    pca.fit_transform(X)
    loadings = getPCALoadings(pca)

    labels_gene = plot_args_dict_sf['feature']['names']

    for d in range(dims):
        loading = loadings[d]
        ranking = np.argsort(loading)[::-1]
        ranking = np.concatenate([ranking[:num_genes],ranking[-num_genes:]])
        loading_ranked = loading[ranking]
        labels_ranked = labels_gene[ranking]

        df = pd.DataFrame({'Genes':labels_ranked,'Contribution':loading_ranked})
        g = sns.barplot(x='Genes',y='Contribution',data=df,**kwargs)
        g.set_xticklabels(labels_ranked,rotation=90,size=5)
        figname = os.path.join(logdir,"Loadings-PCA_{}.pdf".format(d+1))
        plt.tight_layout()
        plt.savefig(figname)
        plt.close()

        ## Save to CSV
        ranking = np.argsort(loading)[::-1]
        loading_ranked = loading[ranking]
        labels_ranked = labels_gene[ranking]
        df = pd.DataFrame({'Genes':labels_ranked,'Contribution':loading_ranked})
        df.to_csv(os.path.join(logdir,'Loadings-Feature-PC_{}.csv'.format(d+1)),sep=',',index=False,header=True)

        ## Perform GSEA
        if genesets is not None:
            for gs_name,geneset in genesets.items():
                outdir = os.path.join(logdir,'GSEA','PCA',gs_name,'dim-{}'.format(d))
                pre_res = gp.prerank(rnk = df, gene_sets = geneset,
                                     outdir = outdir, max_size = 200)


def get_feature_awareness(values_dict,plot_args_dict_sf=None,logdir=None,num_genes=20,genesets=None,**kwargs):

    x_sample = values_dict['model']['X']
    x_recon_base = x_sample.mean(0) # for centered data
    recon_loss_base = np.square(x_sample - x_recon_base)

    decoder_layers = values_dict['model']['decoder_layers']
    DL_sample  = decoder_layers['sample']
    DL_feature = decoder_layers['feature']
    x_recon_list = [np.matmul(score,FL.transpose()) for FL,score in zip(DL_feature,DL_sample)]
    recon_loss_model = np.square(x_sample - x_recon_list)

    FeatureAwareness = -(recon_loss_model - recon_loss_base)

    if plot_args_dict_sf is not None and logdir is not None:

        labels_gene = plot_args_dict_sf['feature']['names']

        for d,FeatAw in enumerate(FeatureAwareness):
            ## Plot
            loading = FeatAw.mean(0)
            ranking = np.argsort(loading)[::-1]
            ranking = np.concatenate([ranking[:num_genes],ranking[-num_genes:]])
            loading_ranked = loading[ranking]
            labels_ranked = labels_gene[ranking]

            df = pd.DataFrame({'Genes':labels_ranked,'Contribution':loading_ranked})
            g = sns.barplot(x='Genes',y='Contribution',data=df,**kwargs)
            _ = g.set_xticklabels(labels_ranked,rotation=90,size=5)
            figname = os.path.join(logdir,"FeatureAwareness-TopFeature-layer_{}.pdf".format(d+1))
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

            ## Save to CSV
            loading = FeatAw.mean(0)
            ranking = np.argsort(loading)[::-1]
            loading_ranked = loading[ranking]
            labels_ranked = labels_gene[ranking]
            df = pd.DataFrame({'Genes':labels_ranked,'Contribution':loading_ranked})
            df.to_csv(os.path.join(logdir,'FA_TopFeature-layer_{}.csv'.format(d+1)),sep=',',index=False,header=True)

            ## Perform GSEA
            if genesets is not None:
                for gs_name,geneset in genesets.items():
                    outdir = os.path.join(logdir,'GSEA','FeatureAwareness',gs_name,'dim-{}'.format(d))
                    pre_res = gp.prerank(rnk = df, gene_sets = geneset,
                                         outdir = outdir, max_size = 200)

    return FeatureAwareness


def save_losses(result_dict,logdir):
    losses = result_dict['model']['losses']
    df = pd.DataFrame(losses)
    df.to_csv(os.path.join(logdir,'losses.csv'),sep=',',index=False,header=True)

    if 'sample' in result_dict.keys():
        losses = result_dict['sample']['losses']
        df = pd.DataFrame(losses)
        df.to_csv(os.path.join(logdir,'losses_sample.csv'),sep=',',index=False,header=True)

    if 'feature' in result_dict.keys():
        losses = result_dict['feature']['losses']
        df = pd.DataFrame(losses)
        df.to_csv(os.path.join(logdir,'losses_feature.csv'),sep=',',index=False,header=True)


def get_feature_awareness_pca(values_dict,plot_args_dict_sf=None,logdir=None,num_genes=20,**kwargs):

    x_sample = values_dict['model']['X']
    x_recon_base = x_sample.mean(0) # for centered data
    recon_loss_base = np.square(x_sample - x_recon_base)
    LE_dim  = values_dict['model']['z_mu'].shape[-1]

    pca = PCA(LE_dim)
    x_pca   = pca.fit_transform(x_sample)
    x_recon = pca.inverse_transform(x_pca)
    recon_loss_model = np.square(x_sample - x_recon)
    FeatureAwareness = -(recon_loss_model - recon_loss_base)

    if plot_args_dict_sf is not None and logdir is not None:

        labels_gene = plot_args_dict_sf['feature']['names']

        ## Plot
        loading = FeatureAwareness.mean(0)
        ranking = np.argsort(loading)[::-1]
        ranking = np.concatenate([ranking[:num_genes],ranking[-num_genes:]])
        loading_ranked = loading[ranking]
        labels_ranked = labels_gene[ranking]
        df = pd.DataFrame({'Genes':labels_ranked,'Contribution':loading_ranked})
        g = sns.barplot(x='Genes',y='Contribution',data=df,**kwargs)
        _ = g.set_xticklabels(labels_ranked,rotation=90,size=5)
        figname = os.path.join(logdir,"FA_TopFeature-PCA_{}.pdf".format(LE_dim))
        plt.tight_layout()
        plt.savefig(figname)
        plt.close()

        ## Save to CSV
        loading = FeatureAwareness.mean(0)
        ranking = np.argsort(loading)[::-1]
        loading_ranked = loading[ranking]
        labels_ranked = labels_gene[ranking]
        df = pd.DataFrame({'Genes':labels_ranked,'Contribution':loading_ranked})
        df.to_csv(os.path.join(logdir,'FA_TopFeature-PCA_{}.csv'.format(LE_dim)),sep=',',index=False,header=True)

    return FeatureAwareness


def plot_latent_embeddings(values_dict, plot_args_dict_sf, palette, method_dim_reds = ['PCA','tSNE'], logdir='', multidimension=False, show_legend=True,**kwargs):

    z_mu = values_dict['model']['z_mu']
    v_mu = values_dict['model']['v_mu']

    X_dict = {'Sample-Wise' : z_mu,
              'Feature-wise': v_mu,
              'Reconstruction' : values_dict['model']['X_recon'],
              'Original'       : values_dict['model']['X']}

    if 'sample' in values_dict.keys():
        X_dict['Sample-Wise-pre'] = values_dict['sample']['z_mu']

    if 'feature' in values_dict.keys():
        X_dict['Feature-Wise-pre'] = values_dict['feature']['z_mu']

    for data_type, X in X_dict.items():

        if data_type in ['Sample-Wise','Sample-Wise-pre','Original','Reconstruction']:
            labels_test = plot_args_dict_sf['sample']['labels']

        else:
            labels_test = plot_args_dict_sf['feature']['labels']

        for method_dim_red in method_dim_reds:
            # dataframe with reduced dimensions
            X_plot, dim_labels = reduce_dimensions(X, reduced_dimension = 2, method = method_dim_red)
            df_plot = pd.concat([pd.DataFrame(X_plot), pd.DataFrame(labels_test)], axis = 1)
            df_plot.columns = dim_labels + ["Label"]
            # Plot
            ax = sns.scatterplot(x = dim_labels[0], y = dim_labels[1], hue = "Label", data = df_plot, palette = palette, **kwargs)
            if not show_legend:
                ax.legend_.remove()
            figname = os.path.join(logdir,"Scatter-{}_{}.pdf".format(data_type,method_dim_red))
            plt.savefig(figname)
            plt.close()

        if multidimension and data_type == 'Sample-Wise' and X.shape[-1] < 5:

            method_dim_red = 'multidimension'
            figname = os.path.join(logdir,"Scatter-{}_{}.pdf".format(data_type,method_dim_red))

            with PdfPages(figname) as pp:
                for i,j in itertools.combinations(range(X.shape[-1]),2):
                    X_plot = X[:,[i,j]]
                    dim_labels = ['dim-{}'.format(i),'dim-{}'.format(j)]
                    df_plot = pd.concat([pd.DataFrame(X_plot), pd.DataFrame(labels_test)], axis = 1)
                    df_plot.columns = dim_labels + ["Label"]
                    # Plot
                    ax = sns.scatterplot(x = dim_labels[0], y = dim_labels[1], hue = "Label", data = df_plot, palette = palette, **kwargs)
                    if not show_legend:
                        ax.legend_.remove()
                    pp.savefig()
                    plt.clf()
            plt.close()

        ax = sns.scatterplot(x = dim_labels[0], y = dim_labels[1], hue = "Label", data = df_plot, palette = palette, **kwargs)
        lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        figname = os.path.join(logdir,"Scatter-legend-{}.pdf".format(data_type))
        plt.savefig(figname, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()


def calculate_PVE(X_in, X_predicted):
    var_orig = np.square(X_in.std(0))
    X_resids = X_in - X_predicted
    var_resid = np.square(X_resids.std(-2))
    PVE = (var_orig.sum(-1) - var_resid.sum(-1)) / var_orig.sum(-1)
    return(PVE)


def compare_loadings(values_dict,logdir,PCA_loadings=None):

    siVAE_loadings = values_dict['model']['v_mu'].T

    dims = siVAE_loadings.shape[0]

    if dims < 5:

        if PCA_loadings is None:
            X = values_dict['model']['X']
            pca = PCA(siVAE_loadings.shape[0])
            pca.fit(X)
            PCA_loadings = getPCALoadings(pca)

        figname = os.path.join(logdir,'loadings_comparison_scatter.pdf')
        with PdfPages(figname) as pp:
            for i,j in itertools.product(range(dims),repeat=2):
                df = pd.DataFrame({'siVAE':siVAE_loadings[i],'PCA':PCA_loadings[j]})
                ax = sns.scatterplot(data=df,x='siVAE',y='PCA')
                plt.title('siVAE_dim-{} vs PC-{}'.format(i+1,j+1))
                pp.savefig()
                plt.clf()
        plt.close()


def compare_FeatureAwareness(FA_siVAEs,FA_pca,logdir):
    figname = os.path.join(logdir,'FA_comparison_scatter.pdf')
    with PdfPages(figname) as pp:
        for ii,FA_siVAE in enumerate(FA_siVAEs):
            df = pd.DataFrame({'siVAE':FA_siVAE.mean(0),'PCA':FA_pca.mean(0)})
            ax = sns.scatterplot(data=df,x='siVAE',y='PCA')
            corr = np.corrcoef(FA_siVAE,FA_pca)[0,1]
            plt.title("siVAE l-{} vs PCA: {:.3f}".format(ii,corr))
            pp.savefig()
            plt.clf()
    plt.close()


def PVE_analysis(values_dict,save_result=False,logdir=""):

    X_orig = values_dict['model']['X']
    var_orig = np.square(X_orig.std(0))
    decoder_layers = values_dict['model']['decoder_layers']
    DL_sample  = decoder_layers['sample']
    DL_feature = decoder_layers['feature']
    PVE_layers = []

    for ii,(FL,score) in enumerate(zip(DL_feature,DL_sample)):
        X_recon = np.matmul(score,FL.transpose())
        PVE_layers.append(calculate_PVE(X_orig,X_recon))

        if ii == 0:
            dims = score.shape[1]
            X_dims = np.array([np.matmul(score[:,[d]],FL[:,[d]].transpose()) for d in range(dims)])
            PVE_dims = calculate_PVE(X_orig,X_dims)

    if save_result:
        np.savetxt(os.path.join(logdir,'PVE_dims.csv'),PVE_dims,delimiter=',')
        np.savetxt(os.path.join(logdir,'PVE_layers.csv'),PVE_layers,delimiter=',')

    return(np.array(PVE_layers),PVE_dims)


def recode_embeddings(values_dict, plot_args_dict_sf, ImageDims, n_pc = 3,
                      logdir = ''):
    ## Create subsets
    labels = plot_args_dict_sf['sample']['labels']
    names = np.unique(labels).tolist()
    masks = [np.isin(labels,ii) for ii in np.unique(labels)]

    score_list = values_dict['model']['decoder_layers']['sample']
    loadings_list = values_dict['model']['decoder_layers']['feature']
    decomposition_dict = {"l_{}".format(l+1): (score,loadings.T) for l,(score,loadings) in enumerate(zip(score_list,loadings_list))}
    X = values_dict['model']['X']

    ## 1. Visualize Recoded Embeddings
    for level, (score,loadings) in decomposition_dict.items():
        if score.shape[-1] >= 5:
            images_list = []
            for mask in masks:
                score_masked = score[mask]

                ## Recode
                pca = PCA(n_components = n_pc).fit(score_masked)
                recoded_score = pca.transform(score_masked)
                recoded_loadings = (pca.components_.T * np.sqrt(pca.explained_variance_)).T
                recoded_pcs = np.matmul(recoded_loadings,loadings)
                images_list.append(recoded_pcs)

            ## Plot
            images_list = np.swapaxes(np.array(images_list),0,1)
            images_list = images_list.reshape([*images_list.shape[:-1],*ImageDims])
            colnames = ['Label'] + ['pc-'+str(ii+1) for ii in range(n_pc)]
            rownames = names
            _ = plot_images(images_list, colnames = colnames, rownames=rownames, cutoff=0.01)
            figname = os.path.join(logdir,'Recoded_embeddings-{}.pdf'.format(level))
            plt.savefig(figname)
            plt.close()

    ## 2. Visualize the changes along the pc-1
    for level, (score,loadings) in decomposition_dict.items():
        if level == "l_1":
            if score.shape[-1] >= 5:
                images_list = []
                for name,mask in zip(names,masks):
                    score_masked = score[mask]
                    X_masked     = X[mask]

                    ## Recode
                    pca = PCA(n_components = n_pc).fit(score_masked)
                    recoded_score = pca.transform(score_masked)
                    recoded_loadings = (pca.components_.T * np.sqrt(pca.explained_variance_)).T
                    recoded_pcs = np.matmul(recoded_loadings,loadings)

                    lim = 1
                    idxrange = np.where(np.all([recoded_score[:,1] > -lim, recoded_score[:,1] < lim], axis=0))[0]
                    idxranked = np.argsort(recoded_score[idxrange,0])
                    idxchosen = np.round(np.linspace(0, len(idxranked) - 1, 10)).astype(int)
                    idx_use_siVAE = idxrange[idxranked[idxchosen]]

                    ## PCA recoding
                    pca = PCA(n_components = n_pc,whiten=True)
                    PCA_score = pca.fit_transform(scale(X_masked))
                    idxrange = np.where(np.all([PCA_score[:,1] > -lim, PCA_score[:,1] < lim], axis=0))[0]
                    idxranked = np.argsort(PCA_score[idxrange,0])
                    idxchosen = np.round(np.linspace(0, len(idxranked) - 1, 10)).astype(int)
                    idx_use_PCA = idxrange[idxranked[idxchosen]]

                    ## Plot
                    images_list = [idx_use_siVAE,idx_use_PCA]
                    images_list = [[X_masked[ii] for ii in idx] for idx in images_list]
                    images_list = [[im.reshape(*ImageDims) for im in images] for images in images_list]
                    images_list = [[images_list[icol][irow] for icol in range(len(images_list))] for irow in range(len(images_list[0]))]
                    rownames = ['siVAE','original']
                    colnames = ['']+['Image-{}'.format(ii+1) for ii in range(len(images_list))]
                    _ = plot_images(np.array(images_list), colnames = colnames, rownames=rownames, cutoff=0.01)
                    figname = os.path.join(logdir,'Recoded_embeddings_label-{}.pdf'.format(name))
                    plt.savefig(figname)
                    plt.close()


def plot_feature_awareness(result_dict,scaler,ImageDims,plot_args_dict_sf,logdir):
    labels = plot_args_dict_sf['sample']['labels']
    result = result_dict['model']
    x_sample = result['sample_dict']['input']
    x_recon  = result['sample_dict']['output']
    x_sample_inversed = scaler.inverse_transform(x_sample)
    x_recon_inversed  = scaler.inverse_transform(x_sample)
    x_sample_image    = x_sample_inversed.reshape(*x_sample_inversed.shape[:-1],*ImageDims)
    x_recon_image     = x_recon_inversed.reshape(*x_recon_inversed.shape[:-1],*ImageDims)

    x_recon_base = 0 # for centered data
    recon_loss_base = np.square(x_sample - x_recon_base) + 1e-3

    recon_loss_list = []
    x_recon_list = result_dict['model']['sample_dict']['output_hl'] + [x_recon]

    recon_loss_list = [np.square(x_sample - x_r) for x_r in x_recon_list]

    recon_loss_image_list = []
    FeatureAwareness_list = []

    for FS in recon_loss_list:
        ##
        FS_image = FS.reshape(len(FS),*ImageDims)
        FS_image = FS_image.sum(-1) # Sum over the color channels
        FS_image = FS_image.reshape(len(FS_image),*ImageDims[:-1],1)
        recon_loss_image_list.append(FS_image)
        ##
        FS_centered = FS - recon_loss_base
        FS_centered_image = FS_centered.reshape(len(FS_centered),*ImageDims)
        FS_centered_image = FS_centered_image.sum(-1) # Sum over the color channels
        FS_centered_image = FS_centered_image.reshape(len(FS_centered_image),*ImageDims[:-1],1)
        FeatureAwareness_list.append(FS_centered_image)

    images_list = [x_sample_image] + FeatureAwareness_list + [x_recon_image]
    recon_names = ['l-'+str(ii+1) for ii,_ in enumerate(FeatureAwareness_list)]
    colnames = ['Label','Original'] + recon_names + ['Reconstruction']
    rownames = np.unique(labels)
    idx_sample = np.array([int(len(x_sample)/len(np.unique(labels)) *ii) for ii,_ in enumerate(np.unique(labels))])
    images_list = [images[idx_sample] for images in images_list]
    _ = plot_images(images_list, colnames = colnames, rownames = rownames, cutoff = 0.1)
    figname = os.path.join(logdir,'FeatureAwareness.pdf')
    plt.savefig(figname)
    plt.close()
