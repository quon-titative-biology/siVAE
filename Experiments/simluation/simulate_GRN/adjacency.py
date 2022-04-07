## 2018/12/02
# Gene Regulatory Network Simulation Using Simple Multivariate Gaussian
# Take input through argparse

## Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import offsetbox
from matplotlib import cm
import seaborn as sns

import random
import numpy as np

import os
import network_class as ntwrk

import argparse
parser          = argparse.ArgumentParser()
network_setup   = parser.add_argument('--network_setup', type = lambda s: [item for item in s.split('_')])
logdir          = parser.add_argument('--logdir')

args = parser.parse_args()
network_setup   = args.network_setup
logdir          = args.logdir

#==========================Gaussian Samples with Network =======================
# Structure
# Structure
netwrk = ntwrk.network()
adj_matrix, network_index = netwrk.create_network(network_setup)
size = netwrk.size

def cov2cor(cov_matrix):
    D    = np.sqrt(np.diag(cov_matrix.diagonal()))
    Dinv = np.linalg.inv(D)
    cor_matrix = np.matmul(np.matmul(Dinv,cov_matrix),Dinv)
    return cor_matrix

adj_matrix  = adj_matrix

np.savetxt(os.path.join(logdir,'adjacency_matrix.csv'), adj_matrix, delimiter=',')
np.savetxt(os.path.join(logdir,'network_index.csv'), network_index, delimiter=',')

# Plot precision_matrix
ax = sns.heatmap(adj_matrix, cmap="Blues")
plt.title("Adjacency Matrix")
plt.savefig(os.path.join(logdir, 'AdjacencyMatrix'))
plt.close()


################################## Custom inputs
netwrk = ntwrk.network()
# network_setup = ['5.B-49-0','50.F-1']
# network_setup = ['50.F-1']
network_setup = ['5.F-50','50.F-1']
adj_matrix, network_index = netwrk.create_network(network_setup)

logdir = 'adj/test2'
os.makedirs(logdir,exist_ok=True)
np.savetxt(os.path.join(logdir,'adjacency_matrix.csv'), adj_matrix, delimiter=',')
np.savetxt(os.path.join(logdir,'network_index.csv'), network_index, delimiter=',')

ax = sns.heatmap(adj_matrix, cmap="Blues")
plt.title("Adjacency Matrix")
plt.savefig(os.path.join(logdir,'AdjacencyMatrix'))
plt.close()
