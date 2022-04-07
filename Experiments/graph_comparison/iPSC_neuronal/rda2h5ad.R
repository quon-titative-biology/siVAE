# Require reticulate
# Use reticulate to save

# library(reticulate)
# library(renv)
# renv::install("reticulate")
# renv::use_python()
# 3

library(reticulate)
sc <- import("scanpy")

days = c('D11')
# celltypes = c('FPP','P_FPP')
celltypes = c('P_FPP')

df.eff = read.table('diff_efficiency_neur.csv',sep=",",header=T)

for (day in days) {

    for (ct in celltypes) {

        load(paste0('rita_preprocessed/',ct,'-D11/nonscaled_residuals.rda'))

        exp_dir = file.path('data/rita_preprocessed/h5ad',day,'experiment',ct)
        dir.create(exp_dir, showWarnings = FALSE, recursive=T)

        df.eff.task =  df.eff[df.eff$donor_id %in% names(tasks_meta),]

        for (eff_type in colnames(df.eff.task)) {
          if (grepl('efficiency',eff_type)) {
            eff = df.eff.task[,eff_type]
            donor_id_ordered = df.eff.task$donor_id[order(eff)]
            donor_id_sel = c(as.character(head(donor_id_ordered,5)),
                             as.character(tail(donor_id_ordered,5)))

            filename = file.path(exp_dir,paste0('lines-',eff_type,'.txt'))
            fileConn<-file(filename)
            writeLines(donor_id_sel, fileConn)
            close(fileConn)
          }
        }

        for (task in names(tasks_meta)) {
            task_dir = file.path(exp_dir,task)
            dir.create(task_dir, showWarnings = FALSE, recursive=T)
            X = tasks_nondownsampled[[task]]
            adata <- sc$AnnData(
                X   = X,
                obs = tasks_meta[[task]],
                var = as.data.frame(colnames(X))
            )
            adata$var_names = colnames(X)
            sc$pp$scale(adata)
            sc$AnnData$write_h5ad(adata,file.path(task_dir,'scaled_data.h5ad'))
        }

    }

}
