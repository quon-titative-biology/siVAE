library(Seurat)
library(reticulate)
library(Rtsne)
library(ggplot2)

GetProcessedData <- function(SeuratObj, scale = F){
  if (scale){
    df = data.frame(SeuratObj@assays$RNA@scale.data)
  } else {
    df = data.frame(GetAssayData(SeuratObj))
  }
  return(df)
}

info = read.table('E-MTAB-7407.sdrf.txt',sep = '\t', header = T)

SourceName= info[,1]
Weeks = info[5]
Gender = info[8]
Organ = info[10]
CD45  = info[,13]

info = info[Organ == 'liver' & dir.exists(file.path(SourceName)),]

data.dir = file.path(info[,1],'GRCh38')
dir.exists(data.dir)

metadata = data.frame()
seurat.object = NULL
for(ii in 1:length(info[,1])){
  source.name = info[,1][[ii]]
  print(source.name)
  metadata.new=read.table(file.path(source.name,paste0(source.name,'.csv')),sep=',',header=T)
  print(dim(metadata.new))
  metadata.new$Gender = info[,8][[ii]]
  metadata.new$GestationWeeks  = info[,5][[ii]]
  metadata.new$CD45 = info[,13][[ii]]
  metadata.new$ExperimentID = info[,1][[ii]]
  metadata.new=metadata.new[,-1]
  metadata = rbind(metadata,metadata.new)

  # reads = Read10X(data.dir = file.path(source.name,'GRCh38'), gene.column = 2, unique.features = TRUE)
  # print(dim(reads))
  # seurat.object.new = CreateSeuratObject(reads)
  # if(is.null(seurat.object)){
  #   seurat.object = seurat.object.new
  # } else {
  #   seurat.object = merge(seurat.object,seurat.object.new)
  # }
}


seurat.object = NormalizeData(seurat.object, scale.factor=1e4)
seurat.object = ScaleData(seurat.object, do.scale=T, do.center=T, verbose = T)
seurat.object <- FindVariableFeatures(seurat.object, nfeatures=2000)

SeuratObj = seurat.object
df = GetAssayData(SeuratObj,slots = 'scale.data')
all.genes = rownames(SeuratObj)
var.genes = VariableFeatures(SeuratObj)
exprmat = df[all.genes%in%var.genes,]

save(exprmat,genes,metadata,file="fetal_liver.Rdata")

write.table(exprmat,file = "fetal_liver.csv",sep=",")
write.table(metadata,file = "metadata.csv",sep=",",rownames=F)
write.table(genes,file = "genes.csv",sep=",",rownames=F,colnames=F)

load("fetal_liver.Rdata")

## Save to python numpy
library(reticulate)

exprmat = as.matrix(exprmat)
genes = as.matrix(rownames(exprmat))
metadata_names = colnames(metadata)
metadata = as.matrix(metadata)

np = import("numpy")
file = paste0("fetal_liver.npz")
np$savez(file, exprmat = exprmat, genes = genes, metadata = metadata,
         metadata_names = metadata_names)

samples = sample(1:dim(exprmat)[[2]],10000)
exprmat.sample  = t(exprmat[,samples])
metadata.sample = metadata[samples,]
weeks = metadata.sample[,4]
weeks = sapply(weeks,function(x){strsplit(x,' ')[[1]][1]})
weeks = as.numeric(weeks)

# tsne <- Rtsne(exprmat.sample,verbose=T)

df = data.frame(tsne.1=tsne$Y[,1],tsne.2=tsne$Y[,2],cell.type=metadata.sample[,2],week=weeks)
ggplot(df,aes(x=tsne.1,y=tsne.2,color=cell.type)) + geom_point(show.legend = FALSE)
ggsave('tsne.pdf')
ggplot(df,aes(x=tsne.1,y=tsne.2,color=week)) + geom_point()
ggsave('tsne_week.pdf')
