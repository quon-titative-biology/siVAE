#### Run Gene Relevance on cell embeddings from siVAE

# args = commandArgs(trailingOnly=TRUE)
# logdir_in = args[1]
logdir_in = 'out/data/siVAE'
logdir_result = 'out/siVAE/gene_relevance'

## Load destiny for gene relevance
library(destiny)

## Save to python numpy
library(reticulate)
library(parallel)
library(abind)

np = import("numpy")

## Import the gene expression matrix
gene_relevance_input_dir = file.path(logdir_in,'gene_relevance_input.npz')
npz = np$load(gene_relevance_input_dir, allow_pickle=TRUE)

exprs  = npz$f$exprs
coords = npz$f$coords
gene_names = npz$f$gene_names
cell_names = npz$f$cell_names

rownames(exprs) = cell_names
colnames(exprs) = gene_names

## Iterate through parameter k
for (k in c(10,100,200,1000,'default')){

  print(k)

  ## Set directory to save the results
  logdir_out = file.path(logdir_result,paste0('K-',k))
  dir.create(logdir_out, recursive=T, showWarnings=F)

  ## Run gene relevance
  if (k == 'default'){
    res = gene_relevance(coords = coords,
                         exprs  = exprs)
  } else {
    res = gene_relevance(coords = coords,
                         exprs  = exprs,
                         k      = as.integer(k))
  }

  ## Save results as rdata
  save(res,file=file.path(logdir_out,'gene_relevance_result.rdata'))
  res_array = res@partials

  ## Save results as numpy object
  genes = colnames(res_array)
  file = file.path(logdir_out,"gene_relevance_result.npz")
  np$savez(file, partials=res_array, genes = genes)

}
