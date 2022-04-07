##
import os

import h5py
import scipy.sparse as sp_sparse
import time
import scanpy
import anndata

## =============================================================================
##                                   Run Code
## =============================================================================

## Upload the h5py
dir_data = "1million_neuronal/1M_neurons_filtered_gene_bc_matrices_h5.h5"

f = h5py.File(dir_data,'r')['mm10']
labels_gene = f['gene_names'][()]
data_shape = f['shape'][()]

dsets = f
matrix = sp_sparse.csc_matrix((dsets['data'], dsets['indices'], dsets['indptr']), shape=dsets['shape'])
matrix = matrix.T
matrix_sample = matrix[:20000]

adata = anndata.AnnData(matrix)
adata.var_name = labels_gene
adata.write('expression.h5ad')

## Split the data to save
logdir_adata_parts = 'AnnDataSplit'
os.makedirs(logdir_adata_parts,exist_ok=True)

sample_per_file = 50000
num_files = int(adata.shape[0]/sample_per_file) + 1

for i_file in range(num_files):
    i_start = sample_per_file * (i_file)
    i_end   = sample_per_file * (i_file + 1)
    X = adata[i_start:i_end]
    X.write('expression-{}-{}.h5ad'.format(i_file+1,num_files))


## Save to tfrecord
start = time.time()
for i_file in range(num_file):
    print("file: {}".format(i_file))
    output_file = "/share/quonlab/wkdir/yonchoi/front_selection_2.0/data/10x_genomics/1million_neuronal/tfrecord/1milN_{}-{}.tfrecord".format(i_file+1,num_file)
    writer = tf.python_io.TFRecordWriter(output_file)
    #
    i_start = sample_per_file * (i_file)
    i_end   = sample_per_file * (i_file + 1)
    for ii in range(i_start,i_end):
        X = matrix[ii].toarray()
        # X = matrix[ii]
        feature = {
            'X': tf.train.Feature(float_list=tf.train.FloatList(value=X.flatten()))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()
    print(time.time() - start)
