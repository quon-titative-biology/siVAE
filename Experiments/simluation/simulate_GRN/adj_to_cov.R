## Run qpgraph

library(qpgraph)

args <- commandArgs(trailingOnly = TRUE)

logdir = args[1]
rho    = as.double(args[2])
print(logdir)
print('rho = ')
print(rho)
data = read.csv(file = paste0(logdir,'/adjacency_matrix.csv'), header = FALSE, sep = ',')
nrow = nrow(data)
adj_matrix <- data.matrix(data)
cov_matrix = matrix(qpG2Sigma(g = adj_matrix, rho = rho), nrow = nrow(adj_matrix))
pre_matrix = solve(cov_matrix)
cor_matrix = cov2cor(cov_matrix)

write.table(pre_matrix, paste0(logdir,'/precsion_matrix.csv'), row.names=FALSE, col.names = FALSE, sep = ',')
write.table(cov_matrix, paste0(logdir,'/covariance_matrix.csv'), row.names=FALSE, col.names = FALSE, sep = ',')
write.table(cor_matrix, paste0(logdir,'/correlation_matrix.csv'), row.names=FALSE, col.names = FALSE, sep = ',')

#####
logdir = 'adj/test3'
rho    = 0.9
