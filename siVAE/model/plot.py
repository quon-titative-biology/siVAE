import os
import time
import logging

## Plot
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

import numpy as np
import pandas as pd


def calculate_vmax_vmin(X, cutoff = 0.01):
    """
    X list or arrays of values to be flattened
    """
    assert (cutoff > 0 and cutoff < 0.5), 'cutoff percentage must be between 0 and 0.5'
    flattened_images = np.array([x.reshape(-1) for x in X]).reshape(-1)
    sorted_val = np.sort(flattened_images)
    vmax = sorted_val[int(len(sorted_val) * (1-cutoff))]
    vmin = sorted_val[int(len(sorted_val) * cutoff)]
    return vmax,vmin


def plot_images(X_plots, colnames = None, rownames = None, vmax = None, vmin = None, cutoff = None, center = 0,
                fs_scale = 2, vmaxmin_type = 'individual', cmap = 'RdBu', **kwargs):
    """
    X_images: col x row x images
    colnames: list of colnames (including the first column)
    rownames: either dict or a list of information for each row
    """
    #
    plot_args = {}
    #
    ncols = len(colnames)
    nrows = len(rownames)
    #
    if cutoff is not None:
        if vmaxmin_type == 'all':
            vmax, vmin = calculate_vmax_vmin(X_plots, cutoff)
            vmaxs = np.tile(vmax,(nrows,ncols))
            vmins = np.tile(vmin,(nrows,ncols))
        elif vmaxmin_type == 'row':
            list_vals = [[X_plots[icol][irow] for icol in range(ncols-1)] for irow in range(nrows)]
            vmax_vmin = [calculate_vmax_vmin(x,cutoff) for x in list_vals]
            vmax_vmin = np.tile(np.array(vmax_vmin),[ncols,1,1])
            vmax_vmin = np.transpose(vmax_vmin,[2,1,0])
            vmaxs,vmins = vmax_vmin
        elif vmaxmin_type == 'col':
            list_vals = [[X_plots[icol][irow] for irow in range(nrows)] for icol in range(ncols-1)]
            vmax_vmin = [calculate_vmax_vmin(x,cutoff) for x in list_vals]
            vmax_vmin = np.tile(np.array(vmax_vmin),[nrows,1,1])
            vmax_vmin = np.transpose(vmax_vmin,[2,0,1])
            vmaxs,vmins = vmax_vmin
        elif vmaxmin_type == 'individual':
            list_vals = [[calculate_vmax_vmin(X_plots[icol][irow],cutoff) for icol in range(ncols-1)] for irow in range(nrows)]
            vmax_vmin = np.moveaxis(np.array(list_vals),2,0)
            vmaxs,vmins = vmax_vmin
        else:
            raise Exception('Input a valid vmaxmin_type -> [all, row, col, individual]')
    #
    fig, axes = plt.subplots(nrows = nrows, ncols=ncols, sharex=True, sharey=True, figsize = (ncols*fs_scale,nrows*fs_scale))
    #
    for irow in range(nrows):
        logging.info("Row: {}/{}".format(irow+1,nrows))
        for icol in range(ncols):
            logging.info("Col: {}/{}".format(icol+1,ncols))
            ax = axes[irow][icol]
            if irow == 0:
                _ = ax.set_title(colnames[icol])
            if icol == 0:
                if isinstance(rownames, dict):
                    pass
                else:
                    label = rownames[irow]
                    _ = ax.text(7,5,'{}'.format(label), horizontalalignment='center',verticalalignment='center')
                    #
                _ = ax.set_axis_off()
            else:
                X_plot = X_plots[icol-1][irow]
                if X_plot.shape[-1] == 1:
                    ## Set Vmax Vmin
                    if cutoff is not None:
                        plot_args['vmax'] = vmaxs[irow][icol-1]
                        plot_args['vmin'] = vmins[irow][icol-1]
                    ##
                    X_plot = X_plot.sum(-1)
                    g = sns.heatmap(X_plot,ax=ax,cmap=cmap,cbar=False,
                                    xticklabels=False,yticklabels=False, center = center,
                                    **plot_args, **kwargs)
                else:
                    _ = ax.imshow(X_plot)
                    _ = ax.set_axis_off()
            ax.set(adjustable='box', aspect='equal')
    #
    return fig, axes


def plot_scatter(X_plots, labels_in, dim_labels, colnames = None, rownames = None, vmax = None, vmin = None, cutoff = None, fs_scale = 2,  **kwargs):
    """
    X_images: col x row x images
    colnames: list of colnames (including the first column)
    rownames: either dict or a list of information for each row
    """
    #
    plot_args = {}
    #
    ncols = len(colnames)
    nrows = len(rownames)
    #
    fig, axes = plt.subplots(nrows = nrows, ncols=ncols, sharex=True, sharey=True, figsize = (ncols*fs_scale,nrows*fs_scale))
    #
    for irow in range(nrows):
        logging.info("Row: {}/{}".format(irow+1,nrows))
        for icol in range(ncols):
            logging.info("Col: {}/{}".format(icol+1,ncols))
            ax = axes[irow][icol]
            if irow == 0:
                _ = ax.set_title(colnames[icol])
            if icol == 0:
                if isinstance(rownames, dict):
                    pass
                else:
                    label = rownames[irow]
                    _ = ax.text(7,5,'{}'.format(label), horizontalalignment='center',verticalalignment='center')
                    #
                _ = ax.set_axis_off()
            else:
                X_plot = X_plots[icol-1][irow]
                df_plot = pd.concat([pd.DataFrame(X_plot), pd.DataFrame(labels_in)], axis = 1)
                df_plot.columns = dim_labels + ["Type"]
                # Plot
                sns.scatterplot(x = dim_labels[0], y = dim_labels[1], hue = "Type", data = df_plot, **kwargs)
                plt.title(data_type)
                pp.savefig()
                plt.clf()
            ax.set(adjustable='box', aspect='equal')
    #
    return fig, axes


def plot_grid(X_plots, fun_plot, colnames = None, rownames = None, vmax = None, vmin = None, cutoff = None, fs_scale = 2, **kwargs):
    """
    X_images: col x row x images
    colnames: list of colnames (including the first column)
    rownames: either dict or a list of information for each row
    """
    #
    plot_args = {}
    #
    ncols = len(colnames)
    nrows = len(rownames)
    #
    fig, axes = plt.subplots(nrows = nrows, ncols=ncols, sharex=True, sharey=True, figsize = (ncols*fs_scale,nrows*fs_scale))
    #
    for irow in range(nrows):
        logging.info("Row: {}/{}".format(irow+1,nrows))
        for icol in range(ncols):
            logging.info("Col: {}/{}".format(icol+1,ncols))
            ax = axes[irow][icol]
            if irow == 0:
                _ = ax.set_title(colnames[icol])
            if icol == 0:
                if isinstance(rownames, dict):
                    pass
                else:
                    label = rownames[irow]
                    _ = ax.text(7,5,'{}'.format(label), horizontalalignment='center',verticalalignment='center')
                    #
                _ = ax.set_axis_off()
            else:
                X_plot = X_plots[icol-1][irow]
                df_plot = pd.concat([pd.DataFrame(X_plot), pd.DataFrame(labels_in)], axis = 1)
                df_plot.columns = dim_labels + ["Type"]
                # Plot
                sns.scatterplot(x = dim_labels[0], y = dim_labels[1], hue = "Type", data = df_plot, **kwargs)
                plt.title(data_type)
                pp.savefig()
                plt.clf()
            ax.set(adjustable='box', aspect='equal')
    #
    return fig, axes
