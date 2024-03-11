import torch_geometric
import torch
from nodeutils import *
from edgeutils import *
from graphutils import *

def f1(i):
    res = []
    if (0 <= i <= 7):
        res.append((i + 1) % 8)
        res.append((i + 2) % 8)
        res.append((i + 6) % 8)
        res.append((i + 7) % 8)
        res.append((i + 1) % 8 + 8)
        res.append((i + 7) % 8 + 8)
    if (8 <= i <= 15):
        res.append((i + 2) % 8 + 8)
        res.append((i + 3) % 8 + 8)
        res.append((i + 5) % 8 + 8)
        res.append((i + 6) % 8 + 8)
        res.append((i + 1) % 8)
        res.append((i + 7) % 8)
    return res

def f2(i):
    res = []
    res.append(4 * (i // 4) + (i + 1) % 4)
    res.append(4 * (i // 4) + (i + 2) % 4)
    res.append(4 * (i // 4) + (i + 3) % 4)
    res.append((i + 4) % 16)
    res.append((i + 8) % 16)
    res.append((i + 12) % 16)
    return res

edges1 = {
    i: f1(i) for i in range(16)
}

edges2 = {
    i: f2(i) for i in range(16)
}

edge_index1 = []
edge_index2 = []

for k in edges1:
    for v in edges1[k]:
        edge_index1.append((k, v))

for k in edges2:
    for v in edges2[k]:
        edge_index2.append((k, v))


edge_index1 = torch.LongTensor(edge_index1).T
edge_index2 = torch.LongTensor(edge_index2).T

graph1 = torch_geometric.data.Data(edge_index=edge_index1)
graph2 = torch_geometric.data.Data(edge_index=edge_index2)

print(count_graphlets(graph1))
print(count_graphlets(graph2))
