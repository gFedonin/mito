import math
import matplotlib.pyplot as plt

path_to_pdb = './pdb/1bgy.pdb1'# './pdb/1occ.pdb1' './pdb/5ara.pdb1'
chain_to_prot = {'C': 'cytb'}# {'A': 'cox1', 'B': 'cox2', 'C': 'cox3'} {'W': 'atp6'}
only_selected_chains = False
only_mitochondria_to_nuclear = True
dist_threshold = 8

out_path = './pdb/'


def parse_pdb():
    chain_to_site_coords = {}
    with open(path_to_pdb, 'r') as f:
        curr_chain = ''
        pos_to_coords = {}
        for line in f.readlines():
            s = line.split()
            if s[0] == 'ATOM' and s[2] == 'CA':
                chain = s[4]
                if chain in chain_to_prot.keys():
                    if curr_chain != '':
                        if chain != curr_chain:
                            chain_to_site_coords[chain_to_prot[curr_chain]] = pos_to_coords
                            curr_chain = chain
                            pos_to_coords = {}
                    else:
                        curr_chain = chain
                    pos_to_coords[int(s[5])] = (float(s[6]), float(s[7]), float(s[8]))
        chain_to_site_coords[chain_to_prot[curr_chain]] = pos_to_coords
    return chain_to_site_coords


def dist(c1, c2):
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 + (c1[2] - c2[2])**2)


def get_neighbors_counts(pos_to_coords):
    pos_to_c = []
    for p, c in pos_to_coords.items():
        pos_to_c.append((p, c))
    neighbors = {}
    for p, c in pos_to_c:
        neighbors[p] = 0
    for i in range(len(pos_to_c)):
        p_i, c_i = pos_to_c[i]
        for j in range(i + 1, len(pos_to_c)):
            p_j, c_j = pos_to_c[j]
            if dist(c_i, c_j) < dist_threshold:
                neighbors[p_i] += 1
                neighbors[p_j] += 1
    return list(neighbors.values())


def main():
    chain_to_site_coords = parse_pdb()
    for prot_name, pos_to_coords in chain_to_site_coords.items():
        print(prot_name)
        plt.clf()
        neighbors = get_neighbors_counts(pos_to_coords)
        plt.title('Histogram of neighbor counts')
        plt.xlabel('Neighbor count')
        plt.ylabel('Number of sites')
        n, bins, patches = plt.hist(neighbors, 18, density=False, facecolor='g', alpha=0.75)
        plt.savefig(out_path + prot_name + '_neighbors_hist.png')


if __name__ == '__main__':
    main()