from Bio.PDB import make_dssp_dict
from Bio.PDB.DSSP import residue_max_acc
from Bio.SeqUtils import seq3


pdb_name = '1be3'
dssp_path = '../dssp/'#'./dssp/1bgy.dssp' './dssp/5ara.dssp' './dssp/1occ.dssp'
surf_racer_path = '../surf_racer/'
out_path_sr = '../surf_racer/burried/' + pdb_name + '.csv'#'./Coloring/buried/1bgy.csv' './Coloring/buried/5ara.csv'
out_path_dssp = '../dssp/' + pdb_name + '.csv'#'./Coloring/buried/1bgy.csv' './Coloring/buried/5ara.csv'
chains = {'C'}#'A', 'B',


ras_threshold = 0.05


def parse_dssp():
    max_asa = residue_max_acc['Miller']
    f = open(out_path_dssp, 'w')
    f.write('chain\tpos\taa\tasa\tacc\tburied\n')
    for chain in chains:
        dssp, keys = make_dssp_dict(dssp_path + pdb_name + '_' + chain + '.dssp')
        for (chainid, resid), dssp_stat in dssp.items():
            if chainid in chains:
                aa = seq3(dssp_stat[0]).upper()
                acc = dssp_stat[2]/max_asa[aa]
                if acc < ras_threshold:
                    burried = '1'
                else:
                    burried = '0'
                f.write('%s\t%d\t%s\t%1.3f\t%1.3f\t%s\n' % (chainid, resid[1], dssp_stat[0], dssp_stat[2], acc, burried))


def parse_surf_racer():
    max_asa = residue_max_acc['Miller']
    chain_to_asa = {}
    for chain in chains:
        pos_to_asa = []
        with open(surf_racer_path + pdb_name + '_' + chain + '_residue.txt', 'r') as f:
            for line in f.readlines():
                s = line.strip().split()
                pos_to_asa.append((s[0], s[1], s[2]))
        chain_to_asa[chain] = pos_to_asa
    with open(out_path_sr, 'w') as f:
        for chain, pos_to_asa in chain_to_asa.items():
            for pos, aa, asa in pos_to_asa:
                acc = float(asa)/max_asa[aa]
                if acc < ras_threshold:
                    f.write("%s\t%s\t%s\t%f\tburried\n" % (chain, pos, asa, acc*100))
                else:
                    f.write("%s\t%s\t%s\t%f\tnon_burried\n" % (chain, pos, asa, acc*100))


if __name__ == '__main__':
    # parse_surf_racer()
    parse_dssp()
