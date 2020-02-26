from os import listdir

from compute_cluster_stats import get_internals_dist, get_external_dist, parse_pdb_Aledo_biopython, dist_aledo, \
    parse_site2pdb

# pdb_id = '1be3'
pdb_id = '1bgy'
# pdb_id = '1occ'
# pdb_id = '5ara'
path_to_pdb = '../pdb/' + pdb_id + '.pdb'
chain_to_prot = {'C': 'cytb', 'O': 'cytb'}
prot_to_chain = {'cytb': ['C', 'O']}
# chain_to_prot = {'A': 'cox1', 'N': 'cox1', 'B': 'cox2', 'O': 'cox2', 'C': 'cox3', 'P': 'cox3'}
# prot_to_chain = {'cox1': ['A', 'N'], 'cox2': ['B', 'O'], 'cox3': ['C', 'P']}
# chain_to_prot = {'W': 'atp6'}
# prot_to_chain = {'atp6': 'W'}

dist_threshold = 4
path_to_pos_matching = '../Coloring/mit.int_gaps/'
path = '../mitochondria.mk_dists/'


def add_distances_to_file(fname, chain_to_site_coords, prot_to_site_map):
    dist_f = dist_aledo
    prot_name = fname[:fname.find('.')].lower()
    if prot_name not in prot_to_chain:
        return
    site2pdb = prot_to_site_map[prot_name]
    chains = prot_to_chain[prot_name]
    # neighbors = {}
    max_pos = 0
    poses = []
    for p, c in chain_to_site_coords[chains[0]].items():
        # neighbors[p] = []
        poses.append(p)
        if p > max_pos:
            max_pos = p

    with open(path + fname + '.dist', 'w') as fout:
        internal_dist = get_internals_dist(chain_to_site_coords[chains[0]], dist_f, max_pos)
        for i in range(1, len(chains)):
            internal_dist += get_internals_dist(chain_to_site_coords[chains[i]], dist_f, max_pos)
        internal_dist /= len(chains)

        external_dist = get_external_dist(prot_name, chain_to_site_coords, prot_to_chain, dist_f, max_pos)

        lines = open(path + fname).readlines()
        fout.write(lines[0].strip() + '\tdist\tint/ext\n')
        for l in lines[1:]:
            s = l.strip().split('\t')
            pdb_pos = site2pdb.get(s[0])
            if pdb_pos is None:
                fout.write(l.strip() + '\t-1\tNA\n')
                continue
            p_i = int(pdb_pos)
            pdb_pos = site2pdb.get(s[1])
            if pdb_pos is None:
                fout.write(l.strip() + '\t-1\tNA\n')
                continue
            p_j = int(pdb_pos)
            d_int = internal_dist[p_i, p_j]
            d_ext = external_dist[p_i, p_j]
            if d_int < d_ext:
                fout.write(l.strip() + '\t' + str(d_int) + '\tinternal\n')
            else:
                fout.write(l.strip() + '\t' + str(d_ext) + '\texternal\n')


if __name__ == '__main__':
    chain_to_site_coords = parse_pdb_Aledo_biopython(pdb_id, path_to_pdb, chain_to_prot)
    prot_to_site_map = parse_site2pdb(prot_to_chain, path_to_pos_matching)
    for fname in listdir(path):
        if fname.endswith('.site_pairs'):
            add_distances_to_file(fname, chain_to_site_coords, prot_to_site_map)
