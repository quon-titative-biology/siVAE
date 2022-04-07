import os
import numpy as np

import scanpy as sc

from sklearn.model_selection import KFold

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--datadir',type=str, default='default')
parser.add_argument('--size_threshold', type=int, default=1000)
args = parser.parse_args()

datadir = args.datadir
size_threshold = args.size_threshold
# datadir = 'data/iPSC_neuronal/D11/experiment/P_FPP'
# datadir = 'P_FPP'

tasks = os.listdir(datadir)
tasks = [t for t in tasks if os.path.isdir(os.path.join(datadir,t)) and t != 'plot']

task_valid = []
sizes = []

for task in tasks:
    logdir_task = os.path.join(datadir,task)
    adata = sc.read_h5ad(os.path.join(logdir_task,'scaled_data.h5ad'))
    size = adata.shape[0]
    if size > size_threshold:
        # get size
        n_fold = int(size/size_threshold)
        # create name
        task_names = [f"{task}-{i}" for i in range(n_fold)]
        task_valid += task_names
        # create indices
        n_keep = n_fold * size_threshold
        idx = np.arange(n_keep)
        np.random.shuffle(idx)
        idx = np.array(np.split(idx,n_fold))
        np.save(os.path.join(logdir_task,'kfold-index.npy'), idx)

np.savetxt(os.path.join(datadir, f'lines-downsampled.txt'),
           task_valid, fmt="%s")
