# pdb_id = '1be3'
from os import makedirs
from os.path import exists

from compute_cluster_stats import dist_aledo

# pdb_id = '1bgy'
# pdb_id = '1occ'
from compute_interface_stat_rand_int_igraph import parse_pdb_Aledo_biopython

pdb_id = '5ara'
path_to_pdb = '../pdb/' + pdb_id + '.pdb'
# chain_to_prot = {'C': 'cytb', 'O': 'cytb'}
# prot_to_chain = {'cytb': ['C', 'O']}
# chain_to_prot = {'A': 'cox1', 'N': 'cox1', 'B': 'cox2', 'O': 'cox2', 'C': 'cox3', 'P': 'cox3'}
# prot_to_chain = {'cox1': ['A', 'N'], 'cox2': ['B', 'O'], 'cox3': ['C', 'P']}
chain_to_prot = {'W': 'atp6'}
prot_to_chain = {'atp6': ['W']}

dist_threshold = 7

out_path = '../res/contact_maps/'


if __name__ == '__main__':
    if not exists(out_path):
        makedirs(out_path)
    chain_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb, False, chain_to_prot)
    chains = [chain for chain in chain_to_prot.keys()]
    prot_name = chain_to_prot[chains[0]]

    max_pos = 0
    poses = []
    for p, c in chain_to_site_coords[chains[0]].items():
        poses.append(p)
        if p > max_pos:
            max_pos = p

    with open(out_path + prot_name + '.neighbors', 'w') as f:
        for p_i, c_i in chain_to_site_coords[chains[0]].items():
            neighbors = []
            for chain, pos_to_coords in chain_to_site_coords.items():
                if chain in chain_to_prot:
                    continue
                for p_j, c_j in pos_to_coords.items():
                    dist = dist_aledo(c_i, c_j)
                    if dist < dist_threshold:
                        neighbors.append(chain + ' ' + str(p_j) + ' ' + str(dist))
            f.write(str(p_i) + '\t' + ','.join(neighbors) + '\n')