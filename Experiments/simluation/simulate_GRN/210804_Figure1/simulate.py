## 2021/08/04
# Gene Regulatory Network Simulation Using Simple Multivariate Gaussian
#

## Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import offsetbox
from matplotlib import cm
import seaborn as sns
import pandas as pd

import random
import numpy as np

import os
import simulate_GRN.network_class as ntwrk

from sklearn.decomposition import PCA

################################## Custom inputs
# netwrk = ntwrk.network()
# # network_setup = ['5.B-49-0','50.F-1']
# # network_setup = ['50.F-1']
# network_setup = ['1.F-4','1.F-2','4.F-1']
# adj_matrix, network_index = netwrk.create_network(network_setup)
#
# logdir = 'adj/test7'
# os.makedirs(logdir,exist_ok=True)
# np.savetxt(os.path.join(logdir,'adjacency_matrix.csv'), adj_matrix, delimiter=',')
# np.savetxt(os.path.join(logdir,'network_index.csv'), network_index, delimiter=',')
#
# ax = sns.heatmap(adj_matrix, cmap="Blues")
# plt.title("Adjacency Matrix")
# plt.savefig(os.path.join(logdir,'AdjacencyMatrix'))
# plt.close()

## Sampled
for _ in [1]:
    sample_size=20
    cov_matrix = np.zeros([10,10])
    cov_matrix[:4,:4] = 0.99
    cov_matrix[4:6,4:6] = 0.99
    #
    for ii in range(0,10):
        cov_matrix[ii,ii]=1
    for ii in range(4,5):
        cov_matrix[ii,ii]=1
    for ii in range(5,10):
        cov_matrix[ii,ii]=0.1
    mean_matrix = np.zeros([cov_matrix.shape[0]])
    sample       = np.random.multivariate_normal(mean_matrix, cov_matrix, sample_size)
    sample2      = np.random.multivariate_normal(mean_matrix[:4], cov_matrix[:4,:4], sample_size)
    sample[:,4:6] = sample2[:,:2] * 0.8
    np.save(os.path.join(logdir,'sample.npy'),sample)
    #
    sample_ = sample[np.argsort(-sample[:,0])]
    sns.heatmap(data=sample_, cmap='RdBu', center=0)
    plt.savefig(os.path.join(logdir, 'Simulated data.svg'))
    plt.close()
    #
    cm = ['Blues']*4 + ['Reds']*2 + ['Greens']*4
    df_ = pd.DataFrame(sample_)
    f, axs = plt.subplots(1, df_.columns.size, gridspec_kw={'wspace': 0})
    for i, (s, a, c) in enumerate(zip(df_.columns, axs, cm)):
        sns.heatmap(np.array([df_[s].values]).T, yticklabels=df_.index,
                    xticklabels=[s], ax=a, cmap=c, cbar=False)
        if i>0:
            a.yaxis.set_ticks([])
    plt.savefig(os.path.join(logdir, 'Simulated data colored.svg'))
    plt.close()
    #
    sns.clustermap(data=sample, cmap='RdBu', center=0)
    plt.savefig(os.path.join(logdir, 'Simulated data clustered.svg'))
    plt.close()
    #
    pca = PCA(2)
    pca.fit(sample)
    X_embedding = pca.transform(sample)
    # X_embedding = X_embedding[:,::-1]
    X_embedding[:,1] = -X_embedding[:,1]
    #
    df_plot = pd.DataFrame(X_embedding,columns=['Dim 1','Dim 2'])
    ax = sns.scatterplot(data=df_plot,x='Dim 1',y='Dim 2',s=200,
                         edgecolor='black',linewidth=1,
                         color='gray')
    xmax = df_plot['Dim 1'].abs().max() * 1.2
    ymax = df_plot['Dim 2'].abs().max() * 1.2
    ax.set_xlim([-xmax,xmax])
    ax.set_ylim([-ymax,ymax])
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    plt.savefig(os.path.join(logdir, 'Cell embeddings.svg'))
    plt.close()
    # Plot cell embeddings with color gradient with respect to gene expression
    df_pca = pd.DataFrame(X_embedding,columns=['Dim 1','Dim 2'])
    df_exp = pd.DataFrame(sample,columns=['G-'+str(i+1) for i in range(10)])
    df_plot = pd.concat([df_pca,df_exp],axis=1)
    for ii in range(1,11):
        if ii < 5:
            cmap='Blues'
        elif ii <7:
            cmap='Oranges'
        else:
            cmap='Greens'
        fig, ax = plt.subplots()
        ax = sns.scatterplot(data=df_plot,x='Dim 1',y='Dim 2',s=200,
                             edgecolor='black',linewidth=1,
                             hue='G-'+str(ii),palette=cmap, ax=ax)
        xmax = df_plot['Dim 1'].abs().max() * 1.2
        ymax = df_plot['Dim 2'].abs().max() * 1.2
        ax.set_xlim([-xmax,xmax])
        ax.set_ylim([-ymax,ymax])
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.get_legend().remove()
        norm = plt.Normalize(df_plot['G-'+str(ii)].min(), df_plot['G-'+str(ii)].max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # ax.figure.colorbar(sm)
        plt.savefig(os.path.join(logdir, 'Cell embeddings_{}.svg'.format(ii)))
        plt.close()
    ## Correlation matrix
    sample_ = np.concatenate([sample[:,:4], np.array([np.random.normal(size=20) for ii in range(4)]).transpose()],axis=1)
    np.savetxt(os.path.join(logdir,'sample.csv'), sample_, delimiter=',')
    X_corr = np.corrcoef(sample_.transpose())
    ax = sns.heatmap(X_corr,cmap='RdBu',center=0)
    plt.savefig(os.path.join(logdir, 'CorrGene.svg'))
    plt.close()
    ## Run on R
    data = read.table('adj/test6/sample.csv',sep=',')
    distance = dist(t(data), diag=T)
    library(dendextend)
    library(gplots)
    hc = hclust(distance, 'ward.D')
    dhc = as.dendrogram(hc)
    dhc <- set(dhc, "branches_lwd", 3)
    dhc.reordered = rev(reorder(dhc,c(10:1)))
    svg("adj/test6/gene_dendrogram.svg")
    plot(dhc.reordered)
    dev.off()

    svg(paste0(heatmap_type,"_dendrogram_reordered.svg"))
    plot(dhc.reordered)
    dev.off()
    ## Dendrogram
    from scipy.cluster import hierarchy
    Z = hierarchy.linkage(sample_.transpose(), 'ward', optimal_ordering=True)
    fig,ax = plt.subplots()
    ax.set_facecolor("white")
    hierarchy.dendrogram(Z, ax=ax, color_threshold=0, above_threshold_color='k')
    plt.tight_layout()
    plt.savefig(os.path.join(logdir,'dendrogram.svg'))
    plt.close()
    ## Adjacency matrix
    adj = np.zeros((10,10))
    np.fill_diagonal(adj,1)
    adj[0,:4] = 1
    adj[:4,0] = 1
    adj[:3,:3] = 1
    ax = sns.heatmap(adj,cmap='RdBu',center=0)
    plt.savefig(os.path.join(logdir, 'AdjGene.svg'))
    plt.close()
    #### Loadings
    X_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    # X_loadings = X_loadings[:,::-1]
    X_loadings[:,1] = -X_loadings[:,1]
    # X_loadings[:4] = X_loadings[:4] + np.random.normal(scale=0.1,size=X_loadings.shape)[:4]
    # X_loadings[:10] = X_loadings[:10] + np.random.normal(scale=0.15,size=X_loadings.shape)[:10]
    X_loadings[:4] = X_loadings[:4] + np.array([[0.05,0.2],
                                             [0,-0.2],
                                             [0.2,0],
                                             [-0.15,0]])
    X_loadings[6:10] = X_loadings[6:10] + np.array([[0,0],
                                                     [0,-0.1],
                                                     [0.1,0],
                                                     [-0.1,0]])
    df_plot = pd.DataFrame(X_loadings,columns=['Dim 1','Dim 2'])
    df_plot['Group'] = ['1']*4+['2']*2+['3']*4
    palette = [sns.color_palette('dark')[0],
               sns.color_palette('pastel')[1],
               sns.color_palette('pastel')[2]]
    ax = sns.scatterplot(data=df_plot,x='Dim 1',y='Dim 2',
                         hue='Group',
                         s=300, edgecolor='black',linewidth=1,
                         palette = palette)
    xmax = df_plot['Dim 1'].abs().max() * 1.6
    ymax = df_plot['Dim 2'].abs().max() * 1.2
    ax.set_xlim([-xmax,xmax])
    ax.set_ylim([-ymax,ymax])
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.get_legend().remove()
    plt.savefig(os.path.join(logdir, 'Loadings.svg') )
    plt.close()
    ## Distance Matrix
    dist = np.array([np.linalg.norm(X_loadings-g,axis=1) for g in X_loadings])
    mask = np.invert(np.isin(np.arange(10),[4,5]))
    dist = dist[mask][:,mask]
    sns.heatmap(data=dist)
    plt.savefig(os.path.join(logdir, 'LoadingsDist.svg'))
    plt.close()
