import numpy as np

class network:
    def __init__(self, size = 1):
        self.size = size
        self.adj_matrix = np.zeros([size,size])
        self.mean_matrix = np.zeros([size])
        self.edges = []
        self.diag_val = 0
        self.precision_matrix = None
        self.covariance_matrix = None
        self.network_list = [[] for x in range(size)]
    #
    def create_simple_network(self, network_sizes = [], overlap = []):
        # Input: list of network sizes, e.g. ['1.F-2','1.B-3-2']
        # Assume that each network contains hub at the lowest index
        # Modifies the adj_matrix and edges
        ii = 0
        while len(network_sizes) > len(overlap):
            overlap.append(0)
        #
        for network_ii, (network_size, overlap) in enumerate(zip(network_sizes, overlap)):
            start = ii
            end = ii + network_size
            edges = self.create_edges([start], range(start + 1, end))
            matrix = self.add_edges(edges)
            ii = end - overlap
            for iii in range(start,end):
                self.network_list[iii].append(network_ii+1)
        return self.adj_matrix
    #
    def create_network(self, network_setup = []):
        # B = branching
        # F = fully-connected
        # network_setup = []
        module_str_list = sum([[ntwrk.split('.')[1]] * int(ntwrk.split('.')[0]) for ntwrk in network_setup],[])
        module_list     = [self.build_module(module_str) for module_str in module_str_list]
        self.network_list = sum([[ii+1] * module.shape[0] for ii,module in enumerate(module_list)],[])
        self.adj_matrix = self.merge_modules(module_list)
        self.size       = self.adj_matrix.shape[0]
        return self.adj_matrix, self.network_list
    #
    # utilities
    def build_module(self, module_str):
        module_type = module_str.split('-')[0]
        if module_type == 'B':
        # B-n_branch-depth
        # branching
            _, n_branch, depth = module_str.split('-')
            n_branch = int(n_branch)
            depth    = int(depth)
            if depth == 0:
                size = n_branch + 1
                final_matrix = np.zeros([size,size])
                final_matrix[:,0] = 1
                final_matrix[0,:] = 1
            else:
                module_list = [self.build_module('-'.join([module_type,str(n_branch),str(depth - 1)]))] * (n_branch+1)
                final_matrix = self.merge_modules(module_list)
                hub_idx = np.arange(1,n_branch+1)*(n_branch+1)
                final_matrix[hub_idx,0] = 1
                final_matrix[0,hub_idx] = 1
        #
        if module_type == 'F':
        # Fully Connected
        # F-Size
            _, size = module_str.split('-')
            size    = int(size)
            final_matrix = np.ones([size,size])
        #
        np.fill_diagonal(final_matrix, 1)
        return final_matrix
    #
    def merge_modules(self, module_list):
        matrix_list = []
        for ii, module in enumerate(module_list):
            zero_list     = [np.zeros([module.shape[0], module_j.shape[1]]) for module_j in module_list]
            zero_list[ii] = module
            matrix_list.append(zero_list)
        return np.bmat(matrix_list)
    #
    def create_edges(self, hubs, nodes):
        # Input: list of hubs and nodes
        # return a 2 x n matrix, where n = number of edges
        # edges created between all hubs to all nodes
        return [(node,hub) for node in nodes for hub in hubs]
    #
    def add_edge(self, edge):
        self.adj_matrix[edge[0]][edge[1]] = 1
        return self.adj_matrix
#
    def add_edges(self, edges):
        for edge in edges:
            matrix_mod = self.add_edge(edge)
            matrix_mod = self.add_edge(np.flip(edge,0))
        return self.adj_matrix
#
    # Matrix
    def update_precision_matrix(self):
        diag = np.diag(np.repeat(self.diag_val, size))
        self.precision_matrix = self.adj_matrix + diag
        return self.precision_matrix
        #
    def update_covariance_matrix(self):
        self.covariance_matrix = np.linalg.inv(self.update_precision_matrix())
        return self.covariance_matrix
#
    # diagonal value calculations
    def set_diag_val(self, diag_val):
        self.diag_val = diag_val
        return self.diag_val
        #
    def increase_diag_val(self, diag_inc):
        self.diag_val += diag_inc
        return self.diag_val
#
    def calucualte_min_diag_val(self, diag_inc, init_val):
        if init_val is 0:
            self.diag_val = diag_inc
        else:
            self.diag_val = init_val
        pre_matrix = self.update_precision_matrix()
        cov_matrix = self.update_covariance_matrix()
        while (not np.all(np.linalg.eigvals(cov_matrix) > 0)) or (np.linalg.det(pre_matrix) is 0):
            self.increase_diag_val(diag_inc)
            pre_matrix = self.update_precision_matrix()
            cov_matrix = self.update_covariance_matrix()
        return self.diag_val
            #
    def sample(self, sample_size):
        # return Gaussian Sample
        _ = self.update_precision_matrix()
        _ = self.update_covariance_matrix()
        return np.random.multivariate_normal(self.mean_matrix, self.covariance_matrix, sample_size)
