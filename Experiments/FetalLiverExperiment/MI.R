library(minet)
library(reticulate)
library(ggplot2)
library(RColorBrewer)

np = import("numpy")

####============================================================================
####                    Set directories
####============================================================================

# args = commandArgs(trailingOnly=TRUE)
# logdir_in = args[1]

datadir = 'out/data/siVAE'
logdir  = 'out/GRN'

####============================================================================
####                             Run GRN
####============================================================================

methods = c('clr','aracne','mrnet')

## Settings
plot_heatmap=TRUE


npz.list = list.files(datadir)
npz.list  = lapply(npz.list,function(x){strsplit(x,'.npz')[[1]]})

for (input.type in npz.list) {

  # Import data
  datadir.in = file.path(datadir,paste0(input.type,'.npz'))
  npz        = np$load(datadir.in,allow_pickle=T)
  mat.exp    = npz$f$exprs
  gene.names = npz$f$gene_names

  logdir.input = file.path(logdir,input.type)

  for (method in methods) {

    ## Set directory to save to
    logdir_method = file.path(logdir.input,method)
    dir.create(logdir_method,showWarnings = F, recursive=T)

    ## Run GRN
    adj_mat = minet(mat.exp, method = method)
    rownames(adj_mat) = gene.names
    colnames(adj_mat) = gene.names

    ## Save the GRN results
    write.table(adj_mat,
      file = file.path(logdir_method,'adjmat.csv'),
      row.names = T,
      col.names = T,
      sep = ','
    )

    ## Plot heatmap
    if (plot_heatmap){
      pdf(file.path(logdir_method,'adjmat.pdf'))
      heatmap(adj_mat, col= colorRampPalette(brewer.pal(8, "Blues"))(25),
              Colv = NA, Rowv = NA)
      dev.off()
    }

  }

}
