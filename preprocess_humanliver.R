library(Seurat)
library(reticulate)
np = import("numpy")
## Save
sc.aiz = readRDS('/share/quonlab/wkdir/yonchoi/data/human_liver/GSE124395_Normalhumanliverdata.RData')

sc.aiz.counts = as.matrix(sc.aiz)
# sc.aiz.zonation = read.table('/home/ucdnjj/lab-data/liver_spatial/Massalha_et_al_data/grun_hepatocytes_zonation.txt')
cluster.info = read.table('/share/quonlab/wkdir/yonchoi/data/human_liver/GSE124395_clusterpartition.txt', skip=1, header=F, stringsAsFactors=F)
colnames(cluster.info) = c("cell", "cluster")

exprmat = sc.aiz.counts
metadata = cluster.info

exprmat = as.matrix(exprmat)
genes = as.matrix(rownames(exprmat))
metadata_names = colnames(metadata)
metadata = as.matrix(metadata)

file = paste0("fetal_liver.npz")
np$savez(file, exprmat = exprmat, genes = genes, metadata = metadata,
         metadata_names = metadata_names)


rownames(cluster.info) = cluster.info$cell

sc.liver_atlas <- CreateSeuratObject(counts = sc.aiz.counts, meta.data=cluster.info, project = "spatial_liver", min.cells = 1)
sc.liver_atlas@meta.data$celltype = "?"
sc.liver_atlas@meta.data$celltype[which(sc.liver_atlas@meta.data$cluster %in% c(11,17,14))] = "Hepatocytes"
sc.liver_atlas@meta.data$cluster[which(sc.liver_atlas@meta.data$cluster %in% c(11,17,14))] = "Hepatocytes"
sc.liver_atlas@meta.data$platform = "Aizarani"
sc.liver_atlas@meta.data$zone = sc.aiz.zonation[,1]
sc.liver_atlas = subset(sc.liver_atlas, cells=colnames(sc.liver_atlas)[which((sc.liver_atlas@meta.data$celltype == "Hepatocytes" & is.na(sc.liver_atlas@meta.data$zone))==F)])
sc.liver_atlas = subset(sc.liver_atlas, cells=colnames(sc.liver_atlas)[which(sc.liver_atlas@meta.data$celltype == "Hepatocytes")])
sc.liver_atlas[["percent.mt"]] <- PercentageFeatureSet(sc.liver_atlas, pattern = "^MT\\.")
sc.liver_atlas <- NormalizeData(sc.liver_atlas, scale.factor=1e4)
sc.liver_atlas <- ScaleData(sc.liver_atlas, do.scale=T, do.center=T, verbose = T, vars.to.regress = c("nCount_RNA", "percent.mt"))
sc.liver_atlas <- FindVariableFeatures(sc.liver_atlas, nfeatures=2000)
