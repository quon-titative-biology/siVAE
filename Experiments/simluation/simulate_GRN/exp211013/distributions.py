import numpy as np
import pandas as pd

import anndata

def simulate_network(network="Gaus",n_group=1,**network_kwargs):
    """
    Input
        Network parameters
        network:
    Output
        anndata of sample x feature
    """
    if network == "Gaus":
        simulate_ntwrk = simulate_network_gaus
    elif network == "binary":
        simulate_ntwrk = simulate_network_binary
    else:
        raise ValueError("Input valid network type [Gaus, binary]")
    #
    adata_list = [simulate_ntwrk(i_group=f'G{i+1}',
                                 **network_kwargs) for i in range(n_group)]
    #
    adata = anndata.concat(adata_list,axis=1)
    obs = pd.concat([adata_.obs for adata_ in adata_list],axis=1)
    adata.obs = obs
    return adata


def simulate_network_binary(sample_size=1000, n_tf=1, group_size=10, rho=0.9, percent_active=0.2, i_group=1, **kwargs):
    """ """
    # sample = np.random.uniform(0,1,[sample_size,n_tf])
    # sample = np.round(sample)
    sample = np.zeros([sample_size,group_size])
    sample[:,:n_tf] = np.random.choice([0,1],[sample_size,n_tf],replace=True)
    #
    min_  = 0.5
    idx_on = np.all(sample[:,:n_tf] > min_, axis=1)
    #
    sample[idx_on,n_tf:] = 1
    # for min_ in np.linspace(-2,2,10)[::-1]:
    #     idx_on = np.all(sample_tf > min_, axis=1)
    #     if idx_on.sum()/sample.shape[0] > percent_active:
    #         break
    var = {}
    var['Group'] = i_group
    var['Type']  = [f'TF'] * n_tf + [f'target'] * (group_size-n_tf)
    var['Name']  = [f'{i_group}-TF'] * n_tf + [f'{i_group}-target'] * (group_size-n_tf)
    var = pd.DataFrame(var)
    state = ['Off'] * sample_size
    state[idx_on] = "On"
    obs = pd.DataFrame({'Cell' : range(sample_size),
                        'State': state})
    adata = anndata.AnnData(sample,obs=obs,var=var)
    adata.var_names_make_unique()
    return adata


def multivariate_normal_conditional(mean=None,var=None,index=None):
    """
    Input
        mean  :
        var   :
        index :
    Output
        mean_ : conditional mean of distribution x1
        var_  : conditional var of distribution x1
    """
    return mean1_


def simulate_network_gaus(sample_size=1000, n_tf=1, group_size=10, rho=0.9, offset=0,
                          percent_active=0.2, threshold=None, i_group=1, logic='and',
                          var1=1,var2=1):
    """
    Input
        sample_size: number of samples
        n_tf: number of tfs
        group_size: number of genes
        rho: correlation between correlated genes
        percent_active: target percentage of cells that should be on
        i_group: name of the group for metadata
    Return
        anndata: X is 2D expression matrix
                 obs: metadata for cells
                 var: metadata for genes
    """
    ## Set up covariance matrices
    cov_matrix1 = np.zeros([group_size,group_size])
    np.fill_diagonal(cov_matrix1,1)
    np.fill_diagonal(cov_matrix1[n_tf:,n_tf:],var1)

    cov_matrix2 = np.ones([group_size,group_size]) * rho
    np.fill_diagonal(cov_matrix2,1)
    cov_matrix2 *= var2

    ## Set mean and sample from multivariate Gaussian
    mean_matrix = np.zeros([cov_matrix1.shape[0]])
    sample1       = np.random.multivariate_normal(mean_matrix, cov_matrix1, sample_size)
    mean_matrix = np.zeros([cov_matrix1.shape[0]]) + offset
    sample2       = np.random.multivariate_normal(mean_matrix, cov_matrix2, sample_size)

    ## For cells meeting activation requirement, change downstream genes
    sample = sample1
    sample_tf = sample[:,:n_tf]
    idx_on = get_active_cells(sample_tf,
                              logic=logic,
                              percent_active=percent_active,
                              threshold = threshold)
    sample[idx_on,n_tf:] = sample2[idx_on,n_tf:]

    ## Define metadata with obs/var
    var = {}
    var['Group'] = i_group
    var['Type']  = ['TF'] * n_tf + ['target'] * (group_size-n_tf)
    var['Name']  = [f'{i_group}-TF'] * n_tf + [f'{i_group}-target'] * (group_size-n_tf)
    var = pd.DataFrame(var)
    state = np.array(['Off'] * sample_size)
    state[idx_on] = "On"
    obs = pd.DataFrame({'Cell' : range(sample.shape[0]),
                        'State': state})

    ## Create anndata
    adata = anndata.AnnData(sample,obs=obs,var=var)

    return adata


def logic_gate(B,logic='and'):
    """
    Input
        B: boolean array indicating active genes (*shape, n_genes)
    Return
        Boolean array indicating active cells (*shape)
    """
    # Logic gates for active cells
    if logic == 'and':
        idx_on = np.all(B, axis=-1)
    elif logic == 'xor':
        idx_on = np.sum(B, axis=-1) == 1
    else:
        raise ValueError("Input valid logic ({})".format(logic))
    return idx_on


def get_active_cells(X, logic = 'and', percent_active=None, threshold = None, n_threshold=200):
    """
    Input
        X: 2D numpy array of expression matrix (n_cell, n_tf)
        logic: Str for logic gate (,)
        percent_active: float for percent of active cells
        threshold: float for threshold expression
    Return
        1D boolean array where True indicates active cells (n_cell,)
    """

    if threshold is None:
        # if threshold not defined loop over possible thresholds
        # Select threshold with percentage of active cells closest to input

        if percent_active is None:
            raise ValueError("Either percent_active or threshold must be defined")

        X_max = X.max()
        X_min = X.min()

        thresholds = np.linspace(X_max, X_min, n_threshold)

        B = np.array([X > t for t in thresholds]) # (n_thresholds, n_cell, n_gene)
        idx_on = logic_gate(B,logic) # (n_thresholds, n_cell)

        percent_active_true = idx_on.sum(-1)/X.shape[0] # (n_cell,)
        percent_active_diff = np.abs(percent_active_true - percent_active) # (,)
        threshold = thresholds[percent_active_diff.argmin()]

    idx_on = logic_gate(X > threshold, logic)

    return idx_on
