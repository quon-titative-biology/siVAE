import os
import time
import logging

## Plots
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns

import pandas as pd
import numpy as np

from sklearn.mixture import GaussianMixture
# from scipy.misc import logsumexp

def logsumexp(x,axis):
    return np.log(np.sum(np.exp(x),axis = axis))


def draw_ellipse(position, covariance, num_sig = 1, ax=None, with_center = True, **kwargs):
    """
    Draw an ellipse with a given position and covariance
    Assume diagonal covariance matrix
    """

    ax = ax or plt.gca()

    angle = 0
    width, height = 2 * np.sqrt(covariance)
    x1, x2 = position

    # Draw the Ellipse
    for nsig in range(1, num_sig + 1):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def plot_gmm(alpha, palette, size=1, labels=None, ax=None, mu = None, var = None, gmm = None, hide_legend = False):

    if labels is None:
        labels = np.repeat('Type',mu.shape[0])

    if ax is None:
        _, ax = plt.subplots()

    if gmm is not None:
        mu = gmm.means_
        var = gmm.covariances_
        if mu is None or var is None:
            raise Exception("Either GaussianMixture or a vector of means and variances must be inputted")

    df_plot = pd.concat([pd.DataFrame(mu), pd.DataFrame(labels)], axis = 1)
    dim_labels = ["dim " + str(ii) for ii in range(1,1+mu.shape[-1])]
    df_plot.columns = dim_labels + ["Type"]

    ## Plot
    # Plot contour lines
    for pos, covar, label in zip(mu, var, labels):
        color = palette[label]
        draw_ellipse(pos, covar, ax = ax, color = color, alpha=alpha)

    # Plot the predicted means
    ax = sns.scatterplot(x = dim_labels[0], y = dim_labels[1], s = size,
                         hue = "Type", palette = palette,
                         data = df_plot, linewidth = 0.01)

    if hide_legend:
        ax.get_legend().remove()

    return ax


class GaussianMixtureCustom(GaussianMixture):
    """ Custom Gaussian Mixture, but assuming diagonal covariances"""

    def __init__(self, means, covariances, **kwargs):

        n_components = means.shape[0]
        weights = np.ones(n_components) / n_components
        precisions = 1 / covariances
        precisions_cholesky = 1. / np.sqrt(covariances)

        super().__init__(n_components = n_components, covariance_type = 'diag', **kwargs)

        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances
        self.precision_ = precisions
        self.precisions_cholesky_ = precisions_cholesky

    def calculate_negative_likelihood(self, samples):
        nll = -logsumexp(self._estimate_weighted_log_prob(samples),1)
        return nll


    def plot(self, alpha, palette, size=1, labels=None, ax=None, names = None, means = None, covariances = None, max_num_plot = 10, fontsize = 0.1, **kwargs):
        """
        names: vector, (n_samples)
            vector of labels for the samples, which defaults to components
        """

        if means is None:
            means = self.means_
        if covariances is None:
            covariances = self.covariances_

        ax = plot_gmm(mu = means, var = covariances, alpha = alpha, palette = palette, size = size, labels = labels, ax = ax, **kwargs)

        if names is not None:

            ranked_index = self.rank_index(means)

            names_ranked = names[ranked_index]
            means_ranked = means[ranked_index]

            names_plot = names_ranked[:max_num_plot]
            means_plot = means_ranked[:max_num_plot]

            for name, (x,y) in zip(names_plot, means_plot):
                plt.text(x,y,name,fontsize=fontsize)

        return ax

    def rank_index(self, samples):
        """
        Parameters
        ------
        samples: array-like, shape (n_samples, n_features)
        names : vector, length n_samples
        """

        nll = self.calculate_negative_likelihood(samples)
        return nll.argsort()



if __name__ == "__main__":

    nll = -logsumexp(gmm._estimate_weighted_log_prob(means),1)
    ranked_index_high2low = nll.argsort()
    sym_names = np.char.decode(f['sym_names'][:])
    sym_names_cc = sym_names[idx_cell_cycle_noise_filtered]
    sym_names_cc_ranked = sym_names_cc[ranked_index_high2low]

    FUCCIseq_gene_names_sig = ['Cdk1', 'Ube2c', 'Top2a', 'Hist1h4e', 'Hist1h4c']
    np.isin(FUCCIseq_gene_names_sig,sym_names_cc)
