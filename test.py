# import pygraphviz as PG
#
# G = PG.AGraph()
# nlist = "A B C D E".split()
# a, b = "A A B", "B C D"
# elist = zip(a.split(), b.split())
#
# G.add_nodes_from(nlist)
# G.add_edges_from(elist)
# G.add_edge('A','A',color='blue')
# G.node_attr.update(color="red", style="filled")
# G.edge_attr.update(color="blue", len="2.0", width="2.0")
#
# G.draw('./test.png', format='png', prog='circo')
import Bio.PDB
from Bio.Alphabet import ThreeLetterProtein

pdb_id = '1bgy'
path_to_pdb = '../pdb/' + pdb_id + '.pdb'
only_selected_chains = True
chain_to_prot = {'C': 'cytb'}
prot_to_chain = {'cytb': 'C'}

chain_to_site_coords = {}
structure = Bio.PDB.PDBParser().get_structure(pdb_id, path_to_pdb)
alphabet = set(aa.upper() for aa in ThreeLetterProtein().letters)
model = structure[0]
for chn in model:
    if only_selected_chains and chn.id not in chain_to_prot:
        continue
    pos_to_coords = {}
    pos = 1
    for residue in chn:
        if residue.resname not in alphabet:
            continue
        r1, r2, r3 = residue.id
        if r1 != ' ':
            continue
        # pos = int(residue.id[1])
        atoms = [tuple(atom.coord) for atom in residue]
        pos_to_coords[pos] = atoms
        pos += 1
    chain_to_site_coords[chn.id] = pos_to_coords