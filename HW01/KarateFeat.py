from nodeutils import *
from edgeutils import *
from graphutils import *
from torch_geometric.datasets import KarateClub

def list_formatter(lst):
    return str(list(map(lambda x: f'{x:.4f}', lst))).replace('\'', '')

def main():
    karate = KarateClub()[0]

    # (1) degree, clustering coefficient for every node
    degrees = count_degrees(karate)
    clustering_coeffs = compute_clustering_coeff(karate)

    # (2) Jaccard’s coefficient and Katz index for every node pair
    jaccard_coeffs = compute_jaccard_coeff(karate)
    katz_index = compute_katz_index(karate, beta=0.1)

    # (3) counting of all connected 3,4-node graphlets 
    graphlets = count_graphlets(karate)
    graphlets_fast = count_graphlets_fast(karate)

    with open('KarateResult.txt', 'w') as f:
        f.write('Degrees of each node:\n')
        for v, deg in enumerate(degrees.detach().cpu().numpy()):
            f.write(f'{v}: {deg}\n')
        
        f.write('\nClustering coeff of each node:\n')
        for v, cc in enumerate(clustering_coeffs.detach().cpu().numpy()):
            f.write(f'{v}: {cc:.8f}\n')
        
        f.write('\nJaccard\'s coeff matrix:\n')
        for ll in jaccard_coeffs.tolist():
            f.write(list_formatter(ll) + '\n')

        f.write('\nKatz index matrix:\n')
        for ll in katz_index.tolist():
            f.write(list_formatter(ll) + '\n')
        
        f.write('\nGraphlets count\n')
        for type_, cnt in enumerate(graphlets):
            f.write(f'G{type_ + 1}: {cnt}\n') 

        f.write('\nGraphlets count (fast)\n')
        for type_, cnt in enumerate(graphlets_fast):
            f.write(f'G{type_ + 1}: {cnt}\n') 

if __name__ == '__main__':
    main()