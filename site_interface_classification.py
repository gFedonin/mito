import sys

from Bio.PDB import make_dssp_dict
from Bio.PDB.DSSP import residue_max_acc
from Bio.SeqUtils import seq3

dssp_path = './dssp/1occ.dssp'#'./dssp/1bgy.dssp' './dssp/5ara.dssp'
out_path = './Coloring/buried/1occ.csv'#'./Coloring/buried/1bgy.csv' './Coloring/buried/5ara.csv'
chains = {'A', 'B', 'C'}

ras_threshold = 0.05


def main():
    dssp, keys = make_dssp_dict(dssp_path)
    max_acc = residue_max_acc['Miller']
    f = open(out_path, 'w')
    # f = sys.stdout
    f.write('chain\tpos\taa\tacc\tburied\n')
    for (chainid, resid), dssp_stat in dssp.items():
        if chainid in chains:
            f.write(chainid)
            f.write('\t')
            f.write(str(resid[1]))
            f.write('\t')
            f.write(dssp_stat[0])
            f.write('\t')
            f.write(str(dssp_stat[2]))
            aa = seq3(dssp_stat[0]).upper()
            if dssp_stat[2]/max_acc[aa] < ras_threshold:
                f.write('\t1\n')
            else:
                f.write('\t0\n')


if __name__ == '__main__':
    main()