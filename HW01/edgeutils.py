import torch
import numpy as np
from torch_geometric.utils import to_dense_adj

def compute_jaccard_coeff(graph):
    num_nodes = graph.num_nodes
    edge_index = graph.edge_index
    jaccard_coeffs = torch.zeros((num_nodes, num_nodes))
    for u in range(num_nodes):
        for v in range(num_nodes):
            neigh_of_u = set(edge_index[1, edge_index[0] == u].detach().cpu().numpy().tolist())
            neigh_of_v = set(edge_index[1, edge_index[0] == v].detach().cpu().numpy().tolist())
            jaccard_coeffs[u, v] = len(neigh_of_u & neigh_of_v) / len(neigh_of_u | neigh_of_v)
    return jaccard_coeffs

def compute_katz_index(graph, beta=0.1):
    num_nodes = graph.num_nodes
    edge_index = graph.edge_index
    adj_matrix = to_dense_adj(edge_index).detach().cpu().numpy()[0]
    I = np.eye(num_nodes)
    return  np.linalg.inv(I - beta * adj_matrix) - I