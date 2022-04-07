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

from sklearn.model_selection import train_test_split
import tensorflow as tf
import random
import numpy as np


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
config.allow_soft_placement = True

from deepexplain.tensorflow import DeepExplain

import network_class as ntwrk

import argparse
parser          = argparse.ArgumentParser()
# Directories
logdir          = parser.add_argument('--logdir')
adjdir          = parser.add_argument('--adjdir')
# Network Set up
network_setup   = parser.add_argument('--network_setup', type = lambda s: [item for item in s.split('_')])
# overlap         = parser.add_argument('--overlap', type = lambda s: [int(item) for item in s.split('_')])
Z_dim           = parser.add_argument("--Z_dim",type = int)
h1_dim          = parser.add_argument('--h1_dim', type = int)
# Hyperparameters
l1_scale        = parser.add_argument('--l1_scale', type = float)
l1_scale_h1     = parser.add_argument('--l1_scale_h1', type = float)
l2_scale        = parser.add_argument('--l2_scale', type = float)
beta            = parser.add_argument('--beta', type = float)
# Training Parameters
mb_size         = parser.add_argument('--mb_size', type = int)
lr              = parser.add_argument('--lr', type = float)
iter            = parser.add_argument('--iter', type = int)
test_size       = parser.add_argument('--test_size', type = float)
random_seed     = parser.add_argument('--random_seed', type = int)
sample_size     = parser.add_argument('--sample_size', type = int)



args = parser.parse_args()
# Directories
logdir          = args.logdir
adjdir          = args.adjdir
# Network Set up
network_setup   = args.network_setup
# overlap         = args.overlap
Z_dim           = args.Z_dim
h1_dim          = args.h1_dim
# Hyperparameters
beta            = args.beta
l1_scale        = args.l1_scale
l1_scale_h1     = args.l1_scale_h1
l2_scale        = args.l2_scale
# Training Parameters
mb_size         = args.mb_size
lr              = args.lr
iter            = args.iter
test_size       = args.test_size
random_seed     = args.random_seed
sample_size     = args.sample_size


## Set random_seed
# The sample stays consistent from the same imported covariance matrix
np.random.seed(1)
# Random seed for training changes
if random_seed < 100:
    random.seed(random_seed)
    tf.set_random_seed(random_seed)

f = open(os.path.join(logdir,'Train log.txt'),'w')

#==========================Gaussian Samples with Network =======================
# Structure
netwrk = ntwrk.network()
adj_matrix, network_index = netwrk.create_network(network_setup)
size = netwrk.siz

def cov2cor(cov_matrix):
    D    = np.sqrt(np.diag(cov_matrix.diagonal()))
    Dinv = np.linalg.inv(D)
    cor_matrix = np.matmul(np.matmul(Dinv,cov_matrix),Dinv)
    return cor_matrix

mean_matrix = np.zeros([size])
adj_matrix  = adj_matrix
cov_matrix  = np.loadtxt(os.path.join(adjdir, 'covariance_matrix.csv'), delimiter=',')
cor_matrix  = cov2cor(cov_matrix)
pre_matrix  = np.linalg.inv(cov_matrix)

# Plot Matrices
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

sns.heatmap(data= np.abs(adj_matrix), ax=ax1, cmap = 'Blues').set_title('Adjacency Matrix')
sns.heatmap(data= np.abs(cov_matrix), ax=ax2, cmap = 'Blues').set_title('Covariance Matrix')
sns.heatmap(data= np.abs(pre_matrix), ax=ax3, cmap = 'Blues').set_title('Precision Matrix')
sns.heatmap(data= np.abs(cor_matrix), ax=ax4, cmap = 'Blues').set_title('Correlation Matrix')

plt.savefig(os.path.join(logdir, 'Heatmaps_true'))
plt.close()

## Sampled
cov_matrix = np.zeros([10,10])
cov_matrix[:4,:4] = 0.8
cov_matrix[4:6,4:6] = 0.8

for ii in range(0,10):
    cov_matrix[ii,ii]=1

for ii in range(4,5):
    cov_matrix[ii,ii]=0.8

for ii in range(5,10):
    cov_matrix[ii,ii]=0.01

sample       = np.random.multivariate_normal(mean_matrix, cov_matrix, sample_size)
sample       = sample / np.std(sample,0)
sample_scale = np.mean(np.abs(sample))
f.write("Sample Scale: %.4f" %sample_scale + '\n')

train_data, test_data = train_test_split(sample, test_size=test_size)

# Plot Sampled Matrix
adj_matrix  = adj_matrix
cov_matrix  = np.cov(sample.transpose())
cor_matrix  = cov2cor(cov_matrix)
pre_matrix  = np.linalg.inv(cov_matrix)

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

sns.heatmap(data= np.abs(adj_matrix), ax=ax1, cmap = 'Blues').set_title('Adjacency Matrix')
sns.heatmap(data= np.abs(cov_matrix), ax=ax2, cmap = 'Blues').set_title('Covariance Matrix')
sns.heatmap(data= np.abs(pre_matrix), ax=ax3, cmap = 'Blues').set_title('Precision Matrix')
sns.heatmap(data= np.abs(cor_matrix), ax=ax4, cmap = 'Blues').set_title('Correlation Matrix')

plt.savefig(os.path.join(logdir, 'Heatmaps_sample') )
plt.close()

sns.heatmap(data=sample, cmap='RdBu', center=0)
plt.savefig(os.path.join(logdir, 'Simulated data.svg'))
plt.close()

from sklearn.decomposition import PCA
pca = PCA(2)
pca.fit(sample)
X_pca = pca.transform(sample)
df_plot = pd.DataFrame(X_pca,columns=['PC-1','PC-2'])

sns.scatterplot(data=df_plot,x='PC-1',y='PC-2')
plt.savefig(os.path.join(logdir, 'PCA_score.svg') )
plt.close()

X_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
df_plot = pd.DataFrame(X_loadings,columns=['PC-1','PC-2'])
df_plot['Group'] = ['1']*4+['2']*2+['3']*4
sns.scatterplot(data=df_plot,x='PC-1',y='PC-2',hue='Group')
plt.savefig(os.path.join(logdir, 'PCA_loadings.svg') )
plt.close()


#===========================VAE=================================================
# Network Parameters
X_dim  = size

# tf Graph input
X     = tf.placeholder('float32', [None, X_dim])
Z     = tf.placeholder('float32', [None, Z_dim])

# Weights
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

weights = {
    'Q1_W':      tf.Variable(xavier_init([X_dim, h1_dim])),
    'Q_mu_W':    tf.Variable(xavier_init([h1_dim, Z_dim])),
    'Q_sigma_W': tf.Variable(xavier_init([h1_dim, Z_dim])),
    'P1_W':      tf.Variable(xavier_init([Z_dim, h1_dim])),
    'P2_W':      tf.Variable(xavier_init([h1_dim, X_dim])),
    'Q1_b':      tf.Variable(tf.zeros(shape=[h1_dim])),
    'Q_mu_b':    tf.Variable(tf.zeros(shape=[Z_dim])),
    'Q_sigma_b': tf.Variable(tf.zeros(shape=[Z_dim])),
    'P1_b':      tf.Variable(tf.zeros(shape=[h1_dim])),
    'P2_b':      tf.Variable(tf.zeros(shape=[X_dim]))
}

# Layers
def model(X):
    h1       = tf.nn.relu(tf.matmul(X, weights['Q1_W']) + weights['Q1_b'])
    Z_mu     = (tf.matmul(h1, weights['Q_mu_W']) + weights['Q_mu_b'])
    Z_logvar = (tf.matmul(h1, weights['Q_sigma_W']) + weights['Q_sigma_b'])
    eps      = tf.random_normal(shape=tf.shape(Z_mu))
    Z_sample = tf.add(Z_mu , tf.exp(Z_logvar / 2) * eps, name = 'Z_sample')
    h2       = tf.nn.relu(tf.matmul(Z_sample, weights['P1_W']) + weights['P1_b'])
    X_out    = tf.matmul(h2,weights['P2_W']) + weights['P2_b']
    h2_mu  = tf.nn.relu(tf.matmul(Z_mu, weights['P1_W']) + weights['P1_b'])
    X_mu    = tf.matmul(h2_mu,weights['P2_W']) + weights['P2_b']
    return(Z_mu, Z_logvar, Z_sample, X_out, X_mu)

Z_mu, Z_logvar, Z_sample, X_out, X_mu = model(X)

for key,value in weights.iteritems():
    print key, value

# Losses
loss_recon    = tf.reduce_sum(tf.square(X_out - X), 1) / X_dim
loss_kl       = beta * 0.5 * tf.reduce_sum(tf.exp(Z_logvar) + Z_mu**2 - 1. - Z_logvar, 1) / X_dim
loss_vae      = tf.reduce_mean(0*loss_kl + loss_recon)
loss_l1       = tf.reduce_sum([tf.reduce_sum(tf.abs(tt) * l1_scale) for _, tt in weights.iteritems()])
loss_l1_h1    = tf.reduce_sum([tf.reduce_sum(tf.abs(tt) * l1_scale_h1) for tt in [weights['Q1_W'], weights['Q1_b']]])
loss_l1_total = tf.add(loss_l1, loss_l1_h1)
loss_l2       = tf.reduce_sum([tf.reduce_sum(tf.square(tt) * l2_scale) for _, tt in weights.iteritems()])
loss_reg      = tf.add(loss_l2, loss_l1_total)
loss_total    = tf.add(loss_vae, loss_reg, name = 'loss_total')
solver        = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss_total)

losses = {
    'KL'   : loss_kl,
    'Recon': loss_recon,
    'VAE'  : loss_vae,
    'L1'   : loss_l1,
    'L1_h1': loss_l1_h1,
    'L2'   : loss_l2,
    'Reg'  : loss_reg,
    'Total': loss_total
}

# Initialize variables
init = tf.global_variables_initializer()

def next_batch(data_in, num):
   # input: idx_all, num, i_epoch
   # output: idx, ranom selection of the data
   return(np.array(random.sample(data_in, num)))

sess = tf.Session(config = config)
sess.run(init)

tf.logging.set_verbosity(tf.logging.FATAL)
report_loss = ['Total','Recon','KL','L1','L1_h1','L2']
for it in range(iter):
    X_mb = next_batch(train_data, mb_size)
    _, loss = sess.run([solver, loss_total], feed_dict={X: X_mb})
    if it % 500 == 0 or it == (iter - 1):
        print '--------Iter: %d---------' % it
        f.write('--------Iter: %d---------' % it + '\n')
        print 'Loss: %.4f' %loss
        f.write('Train Loss: %.4f' %loss + '\n')
        print 'Test Data'
        for loss in report_loss:
            curr_loss = sess.run(losses[loss], feed_dict = {X:test_data})
            line = loss + ': %.4f' %np.mean(curr_loss)
            print line
            f.write(line + '\n')

## Save Errors
curr_losses = []
for loss in report_loss:
    curr_loss = sess.run(losses[loss], feed_dict = {X:test_data})
    curr_losses.append(np.mean(curr_loss))

np.savetxt(os.path.join(logdir,'losses.csv'),
           np.array(curr_losses).reshape(1,-1),
           delimiter = ',',
           header    = str(report_loss).strip("[]").replace("'",""),
           comments   = '')

# Plot Recon Error
X_in = test_data
X_new = sess.run(X_out, feed_dict = {X:X_in})
node_recon_error = np.mean(np.square(X_in - X_new),0)

dict_error = {'Recon_loss': node_recon_error}

fig = plt.figure()
index = np.arange(node_recon_error.shape[0])
bar_width = 1/(len(dict_error)+1.)
opacity = 0.8
i = -1

for key,value in dict_error.iteritems():
    i = i+1
    print(i)
    plot = plt.bar(index + bar_width * i, np.abs(value), bar_width, alpha = opacity, label = key)
    print(key)
    plt.xlabel('Module Labels')
    plt.ylabel('')
    plt.title('Error')
    plt.xticks(index, network_index)
    plt.legend()
    plt.tight_layout()
plt.savefig(os.path.join(logdir, 'recon_error'))
plt.close()

# Baseline
X_in = np.zeros([2,X_dim])
X_new = sess.run(X_out, feed_dict = {X:X_in})[0]
node_recon_error = X_new
dict_error = {'Recon_loss': node_recon_error}

fig = plt.figure()
index = np.arange(node_recon_error.shape[0])
bar_width = 1/(len(dict_error)+1.)
opacity = 0.8
i = -1

for key,value in dict_error.iteritems():
    i = i+1
    print(i)
    plot = plt.bar(index + bar_width * i, value, bar_width, alpha = opacity, label = key)
    print(key)
    plt.xlabel('Module Labels')
    plt.ylabel('')
    plt.title('Reconstructed Value')
    plt.xticks(index, network_index)
    plt.legend()
    plt.tight_layout()
plt.savefig(os.path.join(logdir, 'recon_baseline'))
plt.close()

#================================Analysis=======================================
## overall
def DE(input, target, sample, mask = None, method_DE = ['Saliency Maps', 'Gradient * Input', 'Integrated Gradients', 'DeepLIFT', 'Epsilon-LRP', 'Occlusion']):
    if mask is None:
        mask = np.ones(target.shape[1])
    attributions = {}
    for method in method_DE:
        print(method)
        print(dict_DE[method])
        attributions[method] = de.explain(dict_DE[method], target*mask, input, sample)
    return attributions

def DE_marginal(input, target, sample, method_DE = ['Saliency Maps', 'Gradient * Input', 'Integrated Gradients', 'DeepLIFT', 'Epsilon-LRP', 'Occlusion']):
    marginal = []
    n_node = int(target.shape[1])
    for ii in range(n_node):
        mask = np.arange(n_node) == ii
        marginal.append(DE(input     = input,
                           target    = target,
                           sample    = sample,
                           mask      = mask,
                           method_DE = method_DE))
    return(marginal)

dict_DE = {
    'Saliency Maps'       : 'saliency',
    'Gradient * Input'    : 'grad*input',
    'Integrated Gradients': 'intgrad',
    'Epsilon-LRP'         : 'elrp',
    'DeepLIFT'            : 'deeplift',
    'Occlusion'           : 'occlusion'
}

# method_DE = ['Saliency Maps', 'Gradient * Input', 'Integrated Gradients', 'Epsilon-LRP', 'Occlusion']
#
# ## Plot
# with DeepExplain(session=sess) as de:
#     h1       = tf.nn.relu(tf.add(tf.matmul(X, weights['Q1_W']),weights['Q1_b']))
#     Z_mu     = tf.nn.relu(tf.add(tf.matmul(h1, weights['Q_mu_W']),weights['Q_mu_b']))
#     layers = [X, h1, Z_mu]
#     # We run `explain()` several time to compare different attribution methods
#     for input_ii, target_ii in [(0,2)]:
#         input = layers[input_ii]
#         target = layers[target_ii]
#         sample_DE = sess.run(input, feed_dict = {X: test_data})
#         attributions = DE(input     = input,
#                           target    = target,
#                           sample    = sample_DE,
#                           method_DE = method_DE)
#         # Calculate mean attributions
#         # mean_attributions = {}
#         # for key,value in attributions.iteritems():
#         #     mean_attributions[key] = np.abs(np.mean(value,0))
#         attributions_marginal = DE_marginal(input     = input,
#                                             target    = target,
#                                             sample    = sample_DE,
#                                             method_DE = method_DE)
#         # Barplot with Hierarchical Structures
#         n_node = int(target.shape[1])
#         fig = plt.figure()
#         # for loop around target nodes
#         for ii, attributions_curr in enumerate(attributions_marginal):
#             # Calculat Mean Attribution
#             print(ii)s
#             mean_attributions = {}
#             for key,value in attributions_curr.iteritems():
#                 mean_attributions[key] = np.mean(np.abs(value),0)
#             # Create Barplot with Hierarchical Structures
#             ax = fig.add_subplot(n_node, 1, ii+1)
#             index = np.arange(int(input.shape[1]))
#             bar_width = 1/(len(mean_attributions)+1.)
#             opacity = 0.8
#             i = -1
#             # Subplot per target node
#             for key,value in mean_attributions.iteritems():
#                 i = i+1
#                 print(i)
#                 plot = plt.bar(index + bar_width * i, np.abs(value), bar_width, alpha = opacity, label = key)
#                 print(key)
#                 plt.xlabel('Module Labels')
#                 plt.ylabel('')
#                 plt.title('Z' + str(ii))
#                 if input_ii == 0:
#                     plt.xticks(index, network_index)
#                 else:
#                     plt.xticks(index, index)
#                 plt.legend()
#         plt.tight_layout()
#         plt.savefig(os.path.join(logdir, 'analysis_layers' + str(input_ii)) + '-' + str(target_ii))
#         plt.close()


# method_DE = ['Saliency Maps', 'Integrated Gradients', 'Epsilon-LRP', 'Occlusion']
method_DE = ['Integrated Gradients']

# # Plot heatmaps (X to Xout)
with DeepExplain(session=sess) as de:
    h1       = tf.nn.relu(tf.add(tf.matmul(X, weights['Q1_W']),weights['Q1_b']))
    Z_mu     = tf.nn.relu(tf.add(tf.matmul(h1, weights['Q_mu_W']),weights['Q_mu_b']))
    h2       = tf.nn.relu(tf.add(tf.matmul(Z_mu, weights['P1_W']),weights['P1_b']))
    X_out    = tf.add(tf.matmul(h2,weights['P2_W']),weights['P2_b'])
    #
    input  = X
    target = X_out
    sample_DE = test_data
    attributions = DE(input     = input,
                      target    = target,
                      sample    = sample_DE,
                      method_DE = method_DE)
    # Calculate mean attributions
    # mean_attributions = {}
    # for key,value in attributions.iteritems():
    #     mean_attributions[key] = np.abs(np.mean(value,0))
    fig = plt.figure()
    figlen = len(method_DE)
    for ii, method in enumerate(method_DE):
        attributions_marginal = DE_marginal(input     = input,
                                            target    = target,
                                            sample    = sample_DE,
                                            method_DE = [method])
        mtrx = np.array([attrb[method] for attrb in attributions_marginal])
        mtrx = np.array([np.mean(np.abs(attrb[method]), 0) for attrb in attributions_marginal])
        ax = fig.add_subplot(1, figlen, ii+1)
        g = sns.heatmap(data= mtrx.transpose(), ax=ax, xticklabels = network_index, yticklabels = network_index, cmap = 'Blues')
        g.set_title('Heatmap ' + method + str(curr_losses[0]))
        g.set_xticklabels(g.get_xticklabels(), rotation=90)
        g.set_xlabel('Target')
        g.set_ylabel('Input')
        #
    plt.savefig(os.path.join(logdir, 'heatmap_DE_methods'))
    plt.close()

# # Plot heatmaps (Encoder Structure)
with DeepExplain(session=sess) as de:
    h1       = tf.nn.relu(tf.add(tf.matmul(X, weights['Q1_W']),weights['Q1_b']))
    Z_mu     = tf.nn.relu(tf.add(tf.matmul(h1, weights['Q_mu_W']),weights['Q_mu_b']))
    h2       = tf.nn.relu(tf.add(tf.matmul(Z_mu, weights['P1_W']),weights['P1_b']))
    X_out    = tf.add(tf.matmul(h2,weights['P2_W']),weights['P2_b'])
    #
    layers = [X, h1, Z_mu]
    for input_ii, target_ii in [(0,2),(1,2),(0,1)]:
        input = layers[input_ii]
        target = layers[target_ii]
        sample_DE = sess.run(input, feed_dict = {X: test_data})
        attributions = DE(input     = input,
                          target    = target,
                          sample    = sample_DE,
                          method_DE = method_DE)
        # Calculate mean attributions
        # mean_attributions = {}
        # for key,value in attributions.iteritems():
        #     mean_attributions[key] = np.abs(np.mean(value,0))
        fig = plt.figure()
        figlen = len(method_DE)
        index = np.arange(int(input.shape[1]))
        for ii, method in enumerate(method_DE):
            attributions_marginal = DE_marginal(input     = input,
                                                target    = target,
                                                sample    = sample_DE,
                                                method_DE = [method])
            mtrx = np.array([np.mean(np.abs(attrb[method]), 0) for attrb in attributions_marginal])
            ax = fig.add_subplot(1, figlen, ii+1)
            if input_ii == 0:
                g = sns.heatmap(data= mtrx.transpose(), ax=ax, yticklabels = (network_index), cmap = 'Blues')
            else:
                g = sns.heatmap(data= mtrx.transpose(), ax=ax, cmap = 'Blues')
            g.set_title('Heatmap ' + method)
            #
        plt.savefig(os.path.join(logdir, 'heatmap_DE_methods_'+str(input_ii)+'-'+str(target_ii)))
        plt.close()

method_DE = ['Integrated Gradients']

# # Plot heatmaps (Decoder)
with DeepExplain(session=sess) as de:
    h1       = tf.nn.relu(tf.add(tf.matmul(X, weights['Q1_W']),weights['Q1_b']))
    Z_mu     = tf.nn.relu(tf.add(tf.matmul(h1, weights['Q_mu_W']),weights['Q_mu_b']))
    h2       = tf.nn.relu(tf.add(tf.matmul(Z_mu, weights['P1_W']),weights['P1_b']))
    X_out    = tf.add(tf.matmul(h2,weights['P2_W']),weights['P2_b'])
    #
    input  = Z_mu
    target = X_out
    sample_DE = sess.run(input, feed_dict = {X: test_data})
    attributions = DE(input     = input,
                      target    = target,
                      sample    = sample_DE,
                      method_DE = method_DE)
    # Calculate mean attributions
    # mean_attributions = {}
    # for key,value in attributions.iteritems():
    #     mean_attributions[key] = np.abs(np.mean(value,0))
    fig = plt.figure()
    figlen = len(method_DE)
    for ii, method in enumerate(method_DE):
        attributions_marginal = DE_marginal(input     = input,
                                            target    = target,
                                            sample    = sample_DE,
                                            method_DE = [method])
        mtrx = np.array([np.mean(np.abs(attrb[method]), 0) for attrb in attributions_marginal])
        ax = fig.add_subplot(1, figlen, ii+1)
        g = sns.heatmap(data= mtrx.transpose(), ax=ax, xticklabels = network_index, cmap = 'Blues')
        g.set_title('Heatmap ' + method)
        g.set_xlabel('Target')
        g.set_ylabel('Input')
        #
    plt.savefig(os.path.join(logdir, 'heatmap_DE_methods_decoder'))
    plt.close()


# # Create plot 2
# fig, ax = plt.subplots()
# index = np.arange(size)
# bar_width = 1/(len(attributions)+1.)
# opacity = 0.8
#
# i = -1
# for key,value in mean_attributions.iteritems():
#     i = i+1
#     value2 = value / np.sum(np.abs(value))
#     plot = plt.bar(index + bar_width * i, value2, bar_width, alpha = opacity, label = key)
#     print(key)
#
# plt.xlabel('')
# plt.ylabel('')
# plt.title('')
# plt.xticks(index, network_index)
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(logdir, 'analysis2') )
# plt.close()

# ## Zs
# for ii in range(Z_dim):
#     with DeepExplain(session=sess) as de:
#         # logits,_,_,_ = model(X)
#         h1       = tf.nn.relu(tf.add(tf.matmul(X, weights['Q1_W']),weights['Q1_b']))
#         Z_mu     = tf.nn.relu(tf.add(tf.matmul(h1, weights['Q_mu_W']),weights['Q_mu_b']))
#         # We run `explain()` several time to compare different attribution methods
#         logits = Z_mu
#         mask = np.zeros(Z_dim)
#         mask[ii] = 1
#         masked_y = logits * mask
#         xi = test_data
#         attributions = {
#             # Gradient-based
#             'Saliency maps':        de.explain('saliency', masked_y, X, xi),
#             # 'Gradient * Input':     de.explain('grad*input', masked_y, X, xi),
#             'Integrated Gradients': de.explain('intgrad', masked_y, X, xi),
#             # 'Epsilon-LRP':          de.explain('elrp', masked_y, X, xi),
#             'DeepLIFT (Rescale)':   de.explain('deeplift', masked_y, X, xi),
#             #Perturbation-based
#             # 'Occlusion':      de.explain('occlusion', masked_y, X, xi)
#         }
#         print ('Done')
#
#     mean_attributions = {}
#     for key,value in attributions.iteritems():
#         mean_attributions[key] = np.mean(np.abs(value),0)
#
#     for key,value in mean_attributions.iteritems():
#         print(key)
#         print(value)
#
#     # Create plot
#     fig, ax = plt.subplots()
#     index = np.arange(size)
#     bar_width = 1/(len(attributions)+1.)
#     opacity = 0.8
#
#     i = -1
#     for key,value in mean_attributions.iteritems():
#         i = i+1
#         plot = plt.bar(index + bar_width * i, value, bar_width, alpha = opacity, label = key)
#         print(key)
#
#     plt.xlabel('')
#     plt.ylabel('')
#     plt.title('')
#     plt.xticks(index, network_index)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(logdir, 'analysisZ' + str(ii)) )
#     plt.close()
#
#     # Create plot 2
#     fig, ax = plt.subplots()
#     index = np.arange(size)
#     bar_width = 1/(len(attributions)+1.)
#     opacity = 0.8
#
#     i = -1
#     for key,value in mean_attributions.iteritems():
#         i = i+1
#         value2 = value / np.sum(np.abs(value))
#         plot = plt.bar(index + bar_width * i, value2, bar_width, alpha = opacity, label = key)
#         print(key)
#
#     plt.xlabel('')
#     plt.ylabel('')
#     plt.title('')
#     plt.xticks(index, network_index)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(logdir, 'analysisZ' + str(ii) + '_2') )
#     plt.close()
