from itertools import combinations, product
from nodeutils import count_degrees
import torch_geometric
from torch_geometric.utils import to_dense_adj
import torch
import numpy as np
from collections import Counter
from tqdm import tqdm

def count_graphlets(graph):

    num_nodes = graph.num_nodes
    edge_index = graph.edge_index
    data = torch_geometric.data.Data(num_nodes=num_nodes, edge_index=edge_index)
    num_graphlets = np.zeros(8, dtype=int)

    for ind in tqdm(combinations(range(num_nodes), 3)):
        subgraph = data.subgraph(torch.LongTensor(ind))

        # G1, 2 edges (4 directed) is enough
        if subgraph.edge_index.shape[1] == 4:
            num_graphlets[0] += 1

        # G2, 3 edges (6 directed) is enough
        if subgraph.edge_index.shape[1] == 6:
            num_graphlets[1] += 1
        
    for ind in tqdm(combinations(range(num_nodes), 4)):
        subgraph = data.subgraph(torch.LongTensor(ind))

        degs = count_degrees(subgraph).detach().numpy().tolist()
        deg_counter = Counter(degs)

        # G3, two 1-deg nodes and two 2-deg nodes
        if deg_counter == Counter([1, 1, 2, 2]):
            num_graphlets[2] += 1
        
        # G4, one 3-deg node and three 1-deg nodes
        if deg_counter == Counter([1, 1, 1, 3]):
            num_graphlets[3] += 1
        
        # G5, four 2-deg nodes
        if deg_counter == Counter([2, 2, 2, 2]):
            num_graphlets[4] += 1
        
        # G6, one 3-deg node, two 2-deg nodes and one 1-deg node
        if deg_counter == Counter([1, 2, 2, 3]):
            num_graphlets[5] += 1
        
        # G7, two 3-deg nodes and two 2-deg nodes
        if deg_counter == Counter([2, 2, 3, 3]):
            num_graphlets[6] += 1
        
        # G8, four 3-deg nodes
        if deg_counter == Counter([3, 3, 3, 3]):
            num_graphlets[7] += 1

    return num_graphlets


def count_graphlets_fast(graph):

    num_nodes = graph.num_nodes
    edge_index = graph.edge_index
    data = torch_geometric.data.Data(num_nodes=num_nodes, edge_index=edge_index)
    num_graphlets = np.zeros(8, dtype=int)

    adj_matrix = to_dense_adj(edge_index)[0].detach().cpu().numpy()
    adj_square = adj_matrix @ adj_matrix
    adj_cube = adj_matrix @ adj_square
    I = np.eye(num_nodes)

    # G2: Counting triangles by problem 2
    num_graphlets[1] = np.sum(I * adj_cube) // 6 

    # G1: Enumerate 2-deg nodes and discard triangles
    print('Counting #G1 ...')
    for v in tqdm(range(num_nodes)):
        all_neighs = edge_index[1, edge_index[0] == v]
        num_neighs = len(all_neighs)
        num_graphlets[0] += num_neighs * (num_neighs - 1) // 2
    num_graphlets[0] -= num_graphlets[1] * 3

    # G3: Count (u, v) length-3 path
    print('Counting #G3 ...')
    for u in tqdm(range(num_nodes)):
        all_neighs_u = set(np.where(adj_matrix[u, :] > 0)[0].tolist())
        all_3_neighs_u = set(np.where(adj_cube[u, :] > 0)[0].tolist()) - set([u]) - all_neighs_u
        for v in all_3_neighs_u:
            all_neighs_v = set(np.where(adj_matrix[v, :] > 0)[0].tolist())
            new_all_neighs_u = all_neighs_u - all_neighs_v
            new_all_neighs_v = all_neighs_v - all_neighs_u

            for nu, nv in product(new_all_neighs_u, new_all_neighs_v):
                num_graphlets[2] += adj_matrix[nu, nv]
    num_graphlets[2] //= 2

    # G4: Enumerate numbers of deg-3 centers
    print('Counting #G4 ...')
    for u in tqdm(range(num_nodes)):
        all_neighs_u = set(np.where(adj_matrix[u, :] > 0)[0])
        for v1 in all_neighs_u:
            all_neighs_v1 = set(np.where(adj_matrix[v1, :] > 0)[0])
            new_all_neighs_u = all_neighs_u - all_neighs_v1 - set([v1])
            for (v2, v3) in combinations(new_all_neighs_u, 2):
                num_graphlets[3] += (not adj_matrix[v2, v3])
    num_graphlets[3] //= 3

    # G5: Enumerate diag nodes
    print('Counting #G5 ...')
    for u in tqdm(range(num_nodes)):
        all_neighs_u = set(np.where(adj_matrix[u, :] > 0)[0])
        all_2_neighs_u = set(np.where(adj_square[u, :] > 0)[0]) - set([u]) - all_neighs_u
        for v in all_2_neighs_u:
            all_neighs_v = set(np.where(adj_matrix[v, :] > 0)[0])
            shared_neighs = all_neighs_u & all_neighs_v
            for (w1, w2) in combinations(shared_neighs, 2):
                num_graphlets[4] += (not adj_matrix[w1, w2])
    num_graphlets[4] //= 4

    # G6: Similar to G4
    print('Counting #G6 ...')
    for u in tqdm(range(num_nodes)):
        all_neighs_u = set(np.where(adj_matrix[u, :] > 0)[0])
        for v in all_neighs_u:
            all_neighs_v = set(np.where(adj_matrix[v, :] > 0)[0])
            new_all_neighs_u = all_neighs_u - all_neighs_v - set([v])
            for (w1, w2) in combinations(new_all_neighs_u, 2):
                num_graphlets[5] += adj_matrix[w1, w2]

    # G7: Similar to G5, enumerate deg-1 diag
    print('Counting #G7 ...')
    for u in tqdm(range(num_nodes)):
        all_neighs_u = set(np.where(adj_matrix[u, :] > 0)[0])
        all_2_neighs_u = set(np.where(adj_square[u, :] > 0)[0]) - set([u]) - all_neighs_u
        for v in all_2_neighs_u:
            all_neighs_v = set(np.where(adj_matrix[v, :] > 0)[0])
            shared_neighs = all_neighs_u & all_neighs_v
            for (w1, w2) in combinations(shared_neighs, 2):
                num_graphlets[6] += adj_matrix[w1, w2]
    num_graphlets[6] //= 2

    # G8: Similar to G4, but rather consider shared neighbours
    print('Counting #G8 ...')
    for u in tqdm(range(num_nodes)):
        all_neighs_u = set(np.where(adj_matrix[u, :] > 0)[0])
        for v1 in all_neighs_u:
            all_neighs_v1 = set(np.where(adj_matrix[v1, :] > 0)[0])
            shared_neighs = all_neighs_u & all_neighs_v1
            for (v2, v3) in combinations(shared_neighs, 2):
                num_graphlets[7] += adj_matrix[v2, v3]
    num_graphlets[7] //= 12

    return num_graphlets