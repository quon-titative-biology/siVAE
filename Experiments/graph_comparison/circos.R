library(BioCircos)

adj.dir = "Desktop/Gerald/RitaFetalLiver/central-100/0.01"
file.list = list.files(adj.dir)
file.list = file.list[grep('csv',file.list)]
file.list = file.list[grep('siVAE',file.list)]

meta.edges = read.csv(file.path(adj.dir,'sig_edges.csv'),row.names=1)
sig.edges = rownames(meta.edges[meta.edges$pvalue_bool=='True',])

myGenome = list("MT" = 16569,"Y"=12343)
myGenome = list("MT" = 16569,"Y"=0)

mt_dictionary = list('MT-ATP8'= c(8366,8572-100),
                     'MT-ATP6'= c(8527,9207-50),
                     'MT-CO1'=  c(5904,7445),
                     'MT-CO2'=  c(7586,8269),
                     'MT-CO3'=  c(9207,9990),
                     'MT-CYB'=  c(14747,15887),
                     'MT-ND1'=  c(3307,4262),
                     'MT-ND2'=  c(4470,5511),
                     'MT-ND3'=  c(10059,10404),
                     'MT-ND4L'= c(10470,10766-50),
                     'MT-ND4'=  c(10760,12137),
                     'MT-ND5'=  c(12337,14148-50),
                     'MT-ND6'=  c(14149,14673))
palette = brewer.pal(n = 4, name = "Set2")
gene_colors = c(rep(c(palette[1]),2),
                c('#FFFFFF'),
                rep(c(palette[2]),3),
                c('#FFFFFF'),
                rep(c(palette[3]),1),
                c('#FFFFFF'),
                rep(c(palette[4]),7),
                c('#FFFFFF')
              )

df.mt = data.frame(mt_dictionary, check.names=FALSE)
gene2pos = (df.mt[1,] + df.mt[2,])/2

## =============================================================================
tracklist = BioCircosTracklist()

# ArcTrack
begin=as.list(df.mt[1,])
end=as.list(df.mt[2,])
labels=colnames(df.mt)
arcs_chromosomes=rep(c('MT'), length(begin))
tracklist=tracklist+BioCircosArcTrack("MTgene",arcs_chromosomes,begin,end,
  minRadius = 1.3, maxRadius = 1.4, labels = labels, displayLabel = TRUE)

# LinkTrack
edges = read.csv(file.path(adj.dir,file.list[1]))
edges = edges[edges['MT_count'] == 2,]
links_pos_1 = as.list(gene2pos[as.vector(unlist(edges['node1']))])
links_pos_2 = as.list(gene2pos[as.vector(unlist(edges['node2']))])

len.edges = length(links_pos_1)
links_labels=rep(c(''), len.edges)
links_chromosomes_1 = rep(c('MT'), len.edges)
links_chromosomes_2 = rep(c('MT'), len.edges)

tracklist = tracklist + BioCircosLinkTrack('Edges',
  links_chromosomes_1, links_pos_1, links_pos_1,
  links_chromosomes_2, links_pos_2, links_pos_2,
  maxRadius = 1.3, labels = links_labels,
  color="#FF6666")
  # color=rep(c("#FF6666"),length(links_pos_1)) )

BioCircos(tracklist,genome=myGenome,
  genomeFillColor=c("white","white"),
  displayGenomeBorder=F,
  genomeTickDisplay=F,
  genomeLabelDisplay=F)

# BioCircos(tracklist,genome=myGenome,
#   genomeFillColor="Reds",genomeTickDisplay=False)

# ==============================================================================
# other setup
# filename = file.list[1]

# filename = file.list[5]

# filename = file.list[20]

# filename = file.list[31]

filename = file.list[38]

# filename = file.list[40]

# labels=colnames(df.mt)
labels=c("MT-ATP8","MT-ATP6",
         "Space1",
         "MT-CO1","MT-CO2","MT-CO3",
         "Space2",
         "MT-CYB",
         "Space3",
         "MT-ND1", "MT-ND2", "MT-ND3", "MT-ND4L", "MT-ND4", "MT-ND5", "MT-ND6",
         "Space4")
# myGenome = rep(list(2), length(labels))
myGenome = c(2,2,
             1,
             2,2,2,
             1,
             2,
             1,
             2,2,2,2,2,2,2,
             1
           )
names(myGenome)=labels

tracklist = BioCircosTracklist()

# LinkTrack
edges = read.csv(file.path(adj.dir,filename))
edges = edges[edges['MT_count'] == 2,]

edges_name = paste(unlist(edges['node1']),unlist(edges['node2']),sep="_")
edges['CorrEff'] = edges_name %in% sig.edges
edges['crossmodule'] = substr(edges$node1,1,5) != substr(edges$node2,1,5)

# edges_type = c(TRUE,FALSE)
edges_type = c(FALSE,TRUE)
# colors = c("#FF6666","#52eb34")
# colors = c("#e3e8de","#FF6666")
colors = c("#e52b50","#30bfbf")
# for (ii in 1:length(edges_type)) {
for (ii in c(1,2)) {
  color = colors[[ii]]
  ## Option 1
  # edges.in=edges[edges['CorrEff'] == edges_type[[ii]],]

  ## Option 2
  edges.in=edges[edges$CorrEff == TRUE,]
  edges.in=edges.in[edges.in$crossmodule == edges_type[[ii]],]

  if (dim(edges.in)[[1]] > 0){
  links_chromosomes_1 = as.vector(unlist(edges.in['node1']))
  links_chromosomes_2 = as.vector(unlist(edges.in['node2']))
  links_pos_1 = rep(c(1),length(links_chromosomes_1))
  links_pos_2 = rep(c(1),length(links_chromosomes_2))

  tracklist = tracklist + BioCircosLinkTrack(paste('Edges',edges_type[[ii]]),
    links_chromosomes_1, links_pos_1, links_pos_1,
    links_chromosomes_2, links_pos_2, links_pos_2,
    maxRadius = 1,width = "0.3em", displayAxis=FALSE,
    color=color)
  }
}

BioCircos(tracklist,genome=myGenome,
  genomeFillColor=gene_colors,
  displayGenomeBorder=FALSE,
  genomeTicksDisplay=FALSE,
  genomeLabelDisplay=FALSE,
  yChr =  FALSE)
