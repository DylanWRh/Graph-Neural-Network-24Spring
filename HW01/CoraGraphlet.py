from torch_geometric.datasets import Planetoid
from graphutils import *


def main():
    cora = Planetoid("./", name="Cora")[0]
    print(count_graphlets_fast(cora))
    # takes ~2mins, from #G1 ~ #G8, results are 
    # [47411, 1630, 195625, 1042314, 1536, 53570, 2468, 220]

if __name__ == '__main__':
    main()