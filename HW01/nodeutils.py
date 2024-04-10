import torch
from torch_geometric.utils import to_dense_adj
from itertools import combinations

def count_degrees(graph):
    num_nodes = graph.num_nodes
    edge_index = graph.edge_index
    degrees = torch.zeros(num_nodes, dtype=torch.long)
    for e in range(edge_index.shape[1]):
        src, dst = edge_index[:, e]
        degrees[dst] += 1
    return degrees

def compute_clustering_coeff(graph):
    num_nodes = graph.num_nodes
    edge_index = graph.edge_index
    adj_matrix = to_dense_adj(edge_index)
    clustering_coeffs = torch.zeros(num_nodes)
    for v in range(num_nodes):
        all_neighs = edge_index[1, edge_index[0] == v]
        num_neighs = len(all_neighs)
        if num_neighs < 2:
            continue
        neigh_edge_cnt = 0
        for n1, n2 in combinations(all_neighs, 2):
            neigh_edge_cnt += adj_matrix[0, n1, n2]
        clustering_coeffs[v] = 2 * neigh_edge_cnt / (num_neighs * (num_neighs - 1))
    return clustering_coeffs