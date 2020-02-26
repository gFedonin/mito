import Bio
import numpy as np
from Bio.Alphabet import ThreeLetterProtein
from Bio.PDB import make_dssp_dict
from Bio.PDB.DSSP import residue_max_acc
from Bio.SeqUtils import seq3
from sklearn.externals.joblib import Parallel, delayed

from compute_cluster_stats import dist
# from print_random_graphs_int_igraph import parse_pdb_Aledo_biopython
from compute_interface_stat_rand_int_igraph import parse_pdb_Aledo_biopython

pdb_name = '5ara'#'1be3'  '2occ'  '1bgy'
path_to_pdb = '../pdb/' + pdb_name + '.pdb'
dssp_path = '../dssp/'
chain_to_prot = {'W': 'atp6'}
prot_to_chain = {'atp6': ['W']}
# chain_to_prot = {'A': 'cox1', 'B': 'cox2', 'C': 'cox3'}
# chain_to_prot = {'C': 'cytb'}
# chain_to_prot = {'C': 'cytb', 'O': 'cytb'}
# prot_to_chain = {'cytb': ['C', 'O']}
surf_racer_path = '../surf_racer/'
out_path_sr = '../surf_racer/burried/' + pdb_name + '.csv'#'./Coloring/buried/1bgy.csv' './Coloring/buried/5ara.csv'
#'./Coloring/buried/1bgy.csv' './Coloring/buried/5ara.csv'
# chain = 'C'
chain = 'W'
# out_path_aledo = '../Coloring/cytb_1bgy_Aledo_4ang_CO.csv'
out_path_aledo = '../Coloring/atp6_5ara_Aledo_4ang_fixed.csv'

ras_threshold = 0.05
dist_threshold = 4




def parse_dssp(monomer=True):
    max_asa = residue_max_acc['Miller']
    if monomer:
        f = open(dssp_path + pdb_name + '_' + chain + '.csv', 'w')
    else:
        f = open(dssp_path + pdb_name + '.csv', 'w')
    f.write('chain\tpos\taa\tasa\tacc\tburied\n')
    if monomer:
        dssp, keys = make_dssp_dict(dssp_path + pdb_name + '_' + chain + '.dssp')
    else:
        dssp, keys = make_dssp_dict(dssp_path + pdb_name + '.dssp')
    for (chainid, resid), dssp_stat in dssp.items():
        if monomer and chainid != chain:
            continue
        if chainid not in chain_to_prot:
            continue
        aa = seq3(dssp_stat[0]).upper()
        acc = dssp_stat[2]/max_asa[aa]
        if acc < ras_threshold:
            burried = '1'
        else:
            burried = '0'
        f.write('%s\t%d\t%s\t%1.3f\t%1.3f\t%s\n' % (chainid, resid[1], dssp_stat[0], dssp_stat[2], acc, burried))


def parse_dssp_multichain(monomer=True):
    max_asa = residue_max_acc['Miller']
    chain_str = ''.join(chain_to_prot.keys())
    if monomer:
        f = open(dssp_path + pdb_name + '_' + chain_str + '_solo.csv', 'w')
    else:
        f = open(dssp_path + pdb_name + '_' + chain_str + '_complex.csv', 'w')
    f.write('chain\tpos\taa\tasa\tacc\tburied\n')

    if monomer:
        dssp, keys = make_dssp_dict(dssp_path + pdb_name + '_' + chain_str + '.dssp')
    else:
        dssp, keys = make_dssp_dict(dssp_path + pdb_name + '.dssp')
    for (chainid, resid), dssp_stat in dssp.items():
        if chainid not in chain_to_prot:
            continue
        aa = seq3(dssp_stat[0]).upper()
        acc = dssp_stat[2] / max_asa[aa]
        if acc < ras_threshold:
            burried = '1'
        else:
            burried = '0'
        f.write('%s\t%d\t%s\t%1.3f\t%1.3f\t%s\n' % (chainid, resid[1], dssp_stat[0], dssp_stat[2], acc, burried))

# def parse_surf_racer():
#     max_asa = residue_max_acc['Miller']
#     chain_to_asa = {}
#     pos_to_asa = []
#     with open(surf_racer_path + pdb_name + '_' + chain + '_residue.txt', 'r') as f:
#         for line in f.readlines():
#             s = line.strip().split()
#             pos_to_asa.append((s[0], s[1], s[2]))
#     chain_to_asa[chain] = pos_to_asa
#     with open(out_path_sr, 'w') as f:
#         for chain, pos_to_asa in chain_to_asa.items():
#             for pos, aa, asa in pos_to_asa:
#                 acc = float(asa)/max_asa[aa]
#                 if acc < ras_threshold:
#                     f.write("%s\t%s\t%s\t%f\tburried\n" % (chain, pos, asa, acc*100))
#                 else:
#                     f.write("%s\t%s\t%s\t%f\tnon_burried\n" % (chain, pos, asa, acc*100))



def dist_aledo(heavy_atoms1, heavy_atoms2):
    n1 = len(heavy_atoms1)
    n2 = len(heavy_atoms2)
    return min(dist(heavy_atoms1[i], heavy_atoms2[j]) for i in range(n1) for j in range(n2))


def get_neighbors(pos_to_coords):
    pos_to_c = []
    for p, c in pos_to_coords.items():
        pos_to_c.append((p, c))
    neighbors = {}
    for p, c in pos_to_c:
        neighbors[p] = []
    for i in range(len(pos_to_c)):
        p_i, c_i = pos_to_c[i]
        for j in range(i + 1, len(pos_to_c)):
            p_j, c_j = pos_to_c[j]
            if dist_aledo(c_i, c_j) < dist_threshold:
                neighbors[p_i].append(p_j)
                neighbors[p_j].append(p_i)
    return neighbors


def compute_aledo_classification(skip_intra=True):
    chain_to_site_coords = parse_pdb_Aledo_biopython(pdb_name, path_to_pdb, False, chain_to_prot)
    # prot_to_neighbors = {prot: get_neighbors(coords) for prot, coords in prot_to_site_coords.items()}
    site_coords = chain_to_site_coords[chain]
    pos_to_data = {}
    for line in open(dssp_path + pdb_name + '.csv').readlines()[1:]:
        s = line.strip().split()
        pos = int(s[1])
        aa = s[2]
        ASAc = float(s[3])
        ACCc = float(s[4])
        pos_to_data[pos] = (aa, ASAc, ACCc)
    for line in open(dssp_path + pdb_name + '_' + chain + '.csv').readlines()[1:]:
        s = line.strip().split()
        pos = int(s[1])
        ASAm = float(s[3])
        ACCm = float(s[4])
        aa, ASAc, ACCc = pos_to_data[pos]
        pos_to_data[pos] = (aa, ASAc, ACCc, ASAm, ACCm)
    pos_list = sorted(list(pos_to_data.keys()))
    with open(out_path_aledo, 'w') as fout:
        fout.write('chain\tpos\tAA\tASAc\tACCc\tASAm\tACCm\tInterContact\tIntraContact\tCont\tdASA\tBCEE\n')
        for pos in pos_list:
            aa, ASAc, ACCc, ASAm, ACCm = pos_to_data[pos]
            dASA = ASAm-ASAc
            #Cont
            intra = 0
            inter = 0
            coords = site_coords[pos]
            for p, c in site_coords.items():
                if p - 1 <= pos <= p + 1:
                    continue
                if dist_aledo(coords, c) < dist_threshold:
                    intra += 1
            for ch, pos_to_coords in chain_to_site_coords.items():
                if ch == chain:
                    continue
                for p, c in pos_to_coords.items():
                    if dist_aledo(coords, c) < dist_threshold:
                        inter += 1
            if skip_intra:
                if inter == 0:
                    Cont = 0
                else:
                    Cont = 2
            else:
                if intra == 0:
                    if inter == 0:
                        Cont = 0
                    else:
                        Cont = 2
                else:
                    if inter == 0:
                        Cont = 1
                    else:
                        Cont = 3
            #BCEE
            if ACCm <= ras_threshold:
                BCEE = 'BURIED'
            elif Cont > 0:
                BCEE = 'CONT'
            elif dASA > 0:
                BCEE = 'ENC_interface'
            elif dASA == 0:
                BCEE = 'ENC_noninterf'
            else:
                print('%d BCEE fail: dASA=%1.2f!!!' % (pos, dASA))
            # BURIED -> Residues with ACCm <= 5
            # CONT -> Residues with ACCm > 5 & Cont > 0
            # ENC_interface -> Residues with ACCm > 5 & Cont == 0 & dASA > 0
            # ENC_noninterf -> Residues with ACCm > 5 & Cont == 0 & dASA == 0
            fout.write('%s\t%d\t%s\t%1.2f\t%1.2f\t%1.2f\t%1.2f\t%d\t%d\t%d\t%1.2f\t%s\n' %
                       (chain, pos, aa, ASAc, ACCc, ASAm, ACCm, inter, intra, Cont, dASA, BCEE))


def get_contacts_between_chains(pos_to_coords1, pos_to_coords2):
    pos_to_cont_num = {}
    for p1, c1 in pos_to_coords1.items():
        pos_to_cont_num[p1] = 0
        for p2, c2 in pos_to_coords2.items():
            if dist_aledo(c1, c2) < dist_threshold:
                pos_to_cont_num[p1] += 1
    return pos_to_cont_num


def compute_aledo_classification_multichain(skip_intra=True):
    chain_to_site_coords = parse_pdb_Aledo_biopython(pdb_name, path_to_pdb, False, chain_to_prot)
    # prot_to_neighbors = {prot: get_neighbors(coords) for prot, coords in prot_to_site_coords.items()}
    chain_str = ''.join(chain_to_prot.keys())
    chain_to_pos = {ch: {} for ch in chain_to_prot.keys()}
    pos_list = set()
    for line in open(dssp_path + pdb_name + '_' + chain_str + '_complex.csv').readlines()[1:]:
        s = line.strip().split()
        ch = s[0]
        pos = int(s[1])
        pos_list.add(pos)
        aa = s[2]
        ASAc = float(s[3])
        ACCc = float(s[4])
        chain_to_pos[ch][pos] = (ch, aa, ASAc, ACCc)
    pos_list = list(pos_list)
    pos_list.sort()
    for line in open(dssp_path + pdb_name + '_' + chain_str + '_solo.csv').readlines()[1:]:
        s = line.strip().split()
        ch = s[0]
        pos = int(s[1])
        ASAm = float(s[3])
        ACCm = float(s[4])
        ch, aa, ASAc, ACCc = chain_to_pos[ch][pos]
        chain_to_pos[ch][pos] = (ch, aa, ASAc, ACCc, ASAm, ACCm)
    chains = list(chain_to_pos.keys())

    pos_to_cont_sum_inter = {}
    pos_to_cont_sum_intra = {}
    for ch1 in chains:
        pos_to_coords1 = chain_to_site_coords[ch1]
        pos_to_cont_sum_inter_ch1 = {}
        pos_to_cont_sum_intra_ch1 = {}
        tasks = Parallel(n_jobs=-1)(delayed(get_contacts_between_chains)(pos_to_coords1, pos_to_coords2)
                                    for ch2, pos_to_coords2 in chain_to_site_coords.items())
        iter_taks = iter(tasks)
        for ch2, pos_to_coords2 in chain_to_site_coords.items():
            if ch2 in chain_to_prot:
                # pos_to_cont_num = get_contacts_between_chains(pos_to_coords1, pos_to_coords2)
                pos_to_cont_num = next(iter_taks)
                for p, c in pos_to_cont_num.items():
                    c1 = pos_to_cont_sum_intra_ch1.get(p)
                    if c1 is None:
                        pos_to_cont_sum_intra_ch1[p] = c
                    else:
                        pos_to_cont_sum_intra_ch1[p] = c1 + c
            else:
                # pos_to_cont_num = get_contacts_between_chains(pos_to_coords1, pos_to_coords2)
                pos_to_cont_num = next(iter_taks)
                for p,c in pos_to_cont_num.items():
                    c1 = pos_to_cont_sum_inter_ch1.get(p)
                    if c1 is None:
                        pos_to_cont_sum_inter_ch1[p] = c
                    else:
                        pos_to_cont_sum_inter_ch1[p] = c1 + c
        for p,c in pos_to_cont_sum_inter_ch1.items():
            c1 = pos_to_cont_sum_inter.get(p)
            if c1 is None:
                pos_to_cont_sum_inter[p] = c
            else:
                pos_to_cont_sum_inter[p] = c1 + c
        for p,c in pos_to_cont_sum_intra_ch1.items():
            c1 = pos_to_cont_sum_intra.get(p)
            if c1 is None:
                pos_to_cont_sum_intra[p] = c
            else:
                pos_to_cont_sum_intra[p] = c1 + c
    with open(out_path_aledo, 'w') as fout:
        fout.write('chain\tpos\tAA\tASAc\tACCc\tASAm\tACCm\tInterContact\tIntraContact\tCont\tdASA\tBCEE\n')
        for pos in pos_list:
            ch, aa, ASAc, ACCc, ASAm, ACCm = chain_to_pos[chains[0]][pos]
            dASA = ASAm - ASAc
            for ch in chains[1:]:
                ch, aa_i, ASAc_i, ACCc_i, ASAm_i, ACCm_i = chain_to_pos[ch][pos]
                if ASAm_i - ASAc_i > dASA:
                    dASA = ASAm_i - ASAc_i
                if ACCm_i > ACCm:
                    ACCm = ACCm_i

            #Cont
            intra = pos_to_cont_sum_intra[pos]
            inter = pos_to_cont_sum_inter[pos]
            if skip_intra:
                if inter == 0:
                    Cont = 0
                else:
                    Cont = 2
            else:
                if intra == 0:
                    if inter == 0:
                        Cont = 0
                    else:
                        Cont = 2
                else:
                    if inter == 0:
                        Cont = 1
                    else:
                        Cont = 3
            #BCEE
            if ACCm <= ras_threshold:
                BCEE = 'BURIED'
            elif Cont > 0:
                BCEE = 'CONT'
            elif dASA > 0:
                BCEE = 'ENC_interface'
            elif dASA == 0:
                BCEE = 'ENC_noninterf'
            else:
                print('%d BCEE fail: dASA=%1.2f!!!' % (pos, dASA))
            # BURIED -> Residues with ACCm <= 5
            # CONT -> Residues with ACCm > 5 & Cont > 0
            # ENC_interface -> Residues with ACCm > 5 & Cont == 0 & dASA > 0
            # ENC_noninterf -> Residues with ACCm > 5 & Cont == 0 & dASA == 0
            fout.write('%s\t%d\t%s\t%1.2f\t%1.2f\t%1.2f\t%1.2f\t%d\t%d\t%d\t%1.2f\t%s\n' %
                       (chain, pos, aa, ASAc, ACCc, ASAm, ACCm, inter, intra, Cont, dASA, BCEE))


def residue_dist_Ca(residue_one, residue_two):
    """Returns the C-alpha distance between two residues"""
    diff_vector = residue_one["CA"].coord - residue_two["CA"].coord
    return np.sqrt(np.sum(diff_vector * diff_vector))


def residue_dist_all_heavy(residue_one, residue_two):
    min_dist = np.math.inf
    for atom1 in residue_one:
        for atom2 in residue_two:
            diff_vector = atom1.coord - atom2.coord
            d = np.sqrt(np.sum(diff_vector * diff_vector))
            if d < min_dist:
                min_dist = d
    return min_dist


def compute_aledo_classification_biopython(skip_intra=True):
    structure = Bio.PDB.PDBParser().get_structure(pdb_name, path_to_pdb)
    model = structure[0]
    chain_pdb = model[chain]
    # chain_to_site_coords = parse_pdb_Aledo(path_to_pdb)
    # prot_to_neighbors = {prot: get_neighbors(coords) for prot, coords in prot_to_site_coords.items()}
    # site_coords = chain_to_site_coords[chain]
    pos_to_data = {}
    for line in open(dssp_path + pdb_name + '.csv').readlines()[1:]:
        s = line.strip().split()
        pos = int(s[1])
        aa = s[2]
        ASAc = float(s[3])
        ACCc = float(s[4])
        pos_to_data[pos] = (aa, ASAc, ACCc)
    for line in open(dssp_path + pdb_name + '_' + chain + '.csv').readlines()[1:]:
        s = line.strip().split()
        pos = int(s[1])
        ASAm = float(s[3])
        ACCm = float(s[4])
        aa, ASAc, ACCc = pos_to_data[pos]
        pos_to_data[pos] = (aa, ASAc, ACCc, ASAm, ACCm)
    with open(out_path_aledo, 'w') as fout:
        fout.write('chain\tpos\tAA\tASAc\tACCc\tASAm\tACCm\tInterContact\tIntraContact\tCont\tdASA\tBCEE\n')
        for residue in chain_pdb:
            pos = int(residue.id[1])
            if pos not in pos_to_data.keys():
                continue
            aa, ASAc, ACCc, ASAm, ACCm = pos_to_data[pos]
            dASA = ASAm-ASAc
            #Cont
            intra = 0
            inter = 0
            for r in chain_pdb:
                p = int(r.id[1])
                if p not in pos_to_data.keys():
                    continue
                if p - 1 <= pos <= p + 1:
                    continue
                if residue_dist_all_heavy(residue, r) < dist_threshold:
                    intra += 1
            for ch in model:
                if ch.id == chain:
                    continue
                for r in ch:
                    if residue_dist_all_heavy(residue, r) < dist_threshold:
                        inter += 1
            if skip_intra:
                if inter == 0:
                    Cont = 0
                else:
                    Cont = 2
            else:
                if intra == 0:
                    if inter == 0:
                        Cont = 0
                    else:
                        Cont = 2
                else:
                    if inter == 0:
                        Cont = 1
                    else:
                        Cont = 3
            #BCEE
            if ACCm <= ras_threshold:
                BCEE = 'BURIED'
            elif Cont > 0:
                BCEE = 'CONT'
            elif dASA > 0:
                BCEE = 'ENC_interface'
            elif dASA == 0:
                BCEE = 'ENC_noninterf'
            else:
                print('%d BCEE fail: dASA=%1.2f!!!' % (pos, dASA))
            # BURIED -> Residues with ACCm <= 5
            # CONT -> Residues with ACCm > 5 & Cont > 0
            # ENC_interface -> Residues with ACCm > 5 & Cont == 0 & dASA > 0
            # ENC_noninterf -> Residues with ACCm > 5 & Cont == 0 & dASA == 0
            fout.write('%s\t%d\t%s\t%1.2f\t%1.2f\t%1.2f\t%1.2f\t%d\t%d\t%d\t%1.2f\t%s\n' %
                       (chain, pos, aa, ASAc, ACCc, ASAm, ACCm, inter, intra, Cont, dASA, BCEE))


def compute_residue_stat(residue, chn, model):
    intra = 0
    inter = 0
    for ch in model:
        if ch.id == chn:
            continue
        if ch.id in chain_to_prot:
            for r in ch:
                if residue_dist_all_heavy(residue, r) < dist_threshold:
                    intra += 1
        else:
            for r in ch:
                if residue_dist_all_heavy(residue, r) < dist_threshold:
                    inter += 1
    return intra, inter


def compute_aledo_classification_biopython_cox(skip_intra=False):
    structure = Bio.PDB.PDBParser().get_structure(pdb_name, path_to_pdb)
    model = structure[0]
    for chn, prot in chain_to_prot.items():
        print(prot)
        chain_pdb = model[chn]
        pos_to_data = {}
        for line in open(dssp_path + pdb_name + '.csv').readlines()[1:]:
            s = line.strip().split()
            if s[0] != chn:
                continue
            pos = int(s[1])
            aa = s[2]
            ASAc = float(s[3])
            ACCc = float(s[4])
            pos_to_data[pos] = (aa, ASAc, ACCc)
        for line in open(dssp_path + pdb_name + '_' + chn + '.csv').readlines()[1:]:
            s = line.strip().split()
            pos = int(s[1])
            ASAm = float(s[3])
            ACCm = float(s[4])
            aa, ASAc, ACCc = pos_to_data[pos]
            pos_to_data[pos] = (aa, ASAc, ACCc, ASAm, ACCm)
        with open(out_path_aledo, 'a') as fout:
            fout.write('chain\tpos\tAA\tASAc\tACCc\tASAm\tACCm\tInterContact\tIntraContact\tCont\tdASA\tBCEE\n')
            tasks = Parallel(n_jobs=-1)(delayed(compute_residue_stat)(residue, chn, model) for residue in chain_pdb)
            it = iter(tasks)
            for residue in chain_pdb:
                intra, inter = next(it)
                pos = int(residue.id[1])
                if pos not in pos_to_data.keys():
                    continue
                aa, ASAc, ACCc, ASAm, ACCm = pos_to_data[pos]
                dASA = ASAm-ASAc
                #Cont
                # intra = 0
                # inter = 0
                # for ch in model:
                #     if ch.id == chain:
                #         continue
                #     if ch.id in chain_to_prot:
                #         for r in ch:
                #             if residue_dist_all_heavy(residue, r) < dist_threshold:
                #                 intra += 1
                #     else:
                #         for r in ch:
                #             if residue_dist_all_heavy(residue, r) < dist_threshold:
                #                 inter += 1
                if skip_intra:
                    if inter == 0:
                        Cont = 0
                    else:
                        Cont = 2
                else:
                    if intra == 0:
                        if inter == 0:
                            Cont = 0
                        else:
                            Cont = 2
                    else:
                        if inter == 0:
                            Cont = 1
                        else:
                            Cont = 3
                #BCEE
                if ACCm <= ras_threshold:
                    BCEE = 'BURIED'
                elif Cont > 0:
                    BCEE = 'CONT'
                elif dASA > 0:
                    BCEE = 'ENC_interface'
                elif dASA == 0:
                    BCEE = 'ENC_noninterf'
                else:
                    print('%d BCEE fail: dASA=%1.2f!!!' % (pos, dASA))
                # BURIED -> Residues with ACCm <= 5
                # CONT -> Residues with ACCm > 5 & Cont > 0
                # ENC_interface -> Residues with ACCm > 5 & Cont == 0 & dASA > 0
                # ENC_noninterf -> Residues with ACCm > 5 & Cont == 0 & dASA == 0
                fout.write('%s\t%d\t%s\t%1.2f\t%1.2f\t%1.2f\t%1.2f\t%d\t%d\t%d\t%1.2f\t%s\n' %
                           (chain, pos, aa, ASAc, ACCc, ASAm, ACCm, inter, intra, Cont, dASA, BCEE))


def compute_pos_stat(coords, chn, chain_to_site_coords):
    intra = 0
    inter = 0
    for ch, pos_to_coords in chain_to_site_coords.items():
        if ch == chn:
            continue
        if ch in chain_to_prot:
            for p, c in pos_to_coords.items():
                if dist_aledo(coords, c) < dist_threshold:
                    intra += 1
        else:
            for p, c in pos_to_coords.items():
                if dist_aledo(coords, c) < dist_threshold:
                    inter += 1
    return intra, inter


def compute_aledo_classification_cox(skip_intra=False):
    chain_to_site_coords = parse_pdb_Aledo_biopython()
    # prot_to_neighbors = {prot: get_neighbors(coords) for prot, coords in prot_to_site_coords.items()}
    for chn, prot in chain_to_prot.items():
        print(prot)
        site_coords = chain_to_site_coords[chn]
        pos_to_data = {}
        for line in open(dssp_path + pdb_name + '.csv').readlines()[1:]:
            s = line.strip().split()
            if s[0] != chn:
                continue
            pos = int(s[1])
            aa = s[2]
            ASAc = float(s[3])
            ACCc = float(s[4])
            pos_to_data[pos] = (aa, ASAc, ACCc)
        for line in open(dssp_path + pdb_name + '_' + chn + '.csv').readlines()[1:]:
            s = line.strip().split()
            pos = int(s[1])
            ASAm = float(s[3])
            ACCm = float(s[4])
            aa, ASAc, ACCc = pos_to_data[pos]
            pos_to_data[pos] = (aa, ASAc, ACCc, ASAm, ACCm)
        pos_list = sorted(list(pos_to_data.keys()))
        with open(out_path_aledo, 'a') as fout:
            fout.write('chain\tpos\tAA\tASAc\tACCc\tASAm\tACCm\tInterContact\tIntraContact\tCont\tdASA\tBCEE\n')
            tasks = Parallel(n_jobs=-1)(delayed(compute_pos_stat)(site_coords[pos], chn, chain_to_site_coords)
                                        for pos in pos_list)
            it = iter(tasks)
            for pos in pos_list:
                aa, ASAc, ACCc, ASAm, ACCm = pos_to_data[pos]
                dASA = ASAm-ASAc
                #Cont
                intra, inter = next(it)
                # intra = 0
                # inter = 0
                # coords = site_coords[pos]
                # for ch, pos_to_coords in chain_to_site_coords.items():
                #     if ch == chn:
                #         continue
                #     if ch in chain_to_prot:
                #         for p, c in pos_to_coords.items():
                #             if dist_aledo(coords, c) < dist_threshold:
                #                 intra += 1
                #     else:
                #         for p, c in pos_to_coords.items():
                #             if dist_aledo(coords, c) < dist_threshold:
                #                 inter += 1
                if skip_intra:
                    if inter == 0:
                        Cont = 0
                    else:
                        Cont = 2
                else:
                    if intra == 0:
                        if inter == 0:
                            Cont = 0
                        else:
                            Cont = 2
                    else:
                        if inter == 0:
                            Cont = 1
                        else:
                            Cont = 3
                #BCEE
                if ACCm <= ras_threshold:
                    BCEE = 'BURIED'
                elif Cont > 0:
                    BCEE = 'CONT'
                elif dASA > 0:
                    BCEE = 'ENC_interface'
                elif dASA == 0:
                    BCEE = 'ENC_noninterf'
                else:
                    print('%d BCEE fail: dASA=%1.2f!!!' % (pos, dASA))
                    BCEE = 'ENC_noninterf'
                # BURIED -> Residues with ACCm <= 5
                # CONT -> Residues with ACCm > 5 & Cont > 0
                # ENC_interface -> Residues with ACCm > 5 & Cont == 0 & dASA > 0
                # ENC_noninterf -> Residues with ACCm > 5 & Cont == 0 & dASA == 0
                fout.write('%s\t%d\t%s\t%1.2f\t%1.2f\t%1.2f\t%1.2f\t%d\t%d\t%d\t%1.2f\t%s\n' %
                           (chn, pos, aa, ASAc, ACCc, ASAm, ACCm, inter, intra, Cont, dASA, BCEE))


def compute_aledo_classification_8ang(skip_intra=True):
    prot_to_site_coords = parse_pdb_Aledo_biopython(pdb_name, path_to_pdb, True, chain_to_prot)
    prot = chain_to_prot[chain]
    # prot_to_neighbors = {prot: get_neighbors(coords) for prot, coords in prot_to_site_coords.items()}
    site_coords = prot_to_site_coords[chain_to_prot[chain]]
    pos_to_data = {}
    for line in open(dssp_path + pdb_name + '.csv').readlines()[1:]:
        s = line.strip().split()
        pos = int(s[1])
        aa = s[2]
        ASAc = float(s[3])
        ACCc = float(s[4])
        pos_to_data[pos] = (aa, ASAc, ACCc)
    for line in open(dssp_path + pdb_name + '_' + chain + '.csv').readlines()[1:]:
        s = line.strip().split()
        pos = int(s[1])
        ASAm = float(s[3])
        ACCm = float(s[4])
        aa, ASAc, ACCc = pos_to_data[pos]
        pos_to_data[pos] = (aa, ASAc, ACCc, ASAm, ACCm)
    pos_list = sorted(list(pos_to_data.keys()))
    with open(out_path_aledo, 'w') as fout:
        fout.write('chain\tpos\tAA\tASAc\tACCc\tASAm\tACCm\tInterContact\tIntraContact\tCont\tdASA\tBCEE\n')
        for pos in pos_list:
            aa, ASAc, ACCc, ASAm, ACCm = pos_to_data[pos]
            dASA = ASAm-ASAc
            #Cont
            intra = 0
            inter = 0
            coords = site_coords[pos]
            for p, c in site_coords.items():
                if p - 1 <= pos <= p + 1:
                    continue
                if dist(coords, c) < dist_threshold:
                    intra += 1
            for pr, pos_to_coords in prot_to_site_coords.items():
                if pr == prot:
                    continue
                for p, c in pos_to_coords.items():
                    if dist(coords, c) < dist_threshold:
                        inter += 1
            if skip_intra:
                if inter == 0:
                    Cont = 0
                else:
                    Cont = 2
            else:
                if intra == 0:
                    if inter == 0:
                        Cont = 0
                    else:
                        Cont = 2
                else:
                    if inter == 0:
                        Cont = 1
                    else:
                        Cont = 3
            #BCEE
            if ACCm <= ras_threshold:
                BCEE = 'BURIED'
            elif Cont > 0:
                BCEE = 'CONT'
            elif dASA > 0:
                BCEE = 'ENC_interface'
            elif dASA == 0:
                BCEE = 'ENC_noninterf'
            else:
                print('%d BCEE fail: dASA=%1.2f!!!' % (pos, dASA))
            # BURIED -> Residues with ACCm <= 5
            # CONT -> Residues with ACCm > 5 & Cont > 0
            # ENC_interface -> Residues with ACCm > 5 & Cont == 0 & dASA > 0
            # ENC_noninterf -> Residues with ACCm > 5 & Cont == 0 & dASA == 0
            fout.write('%s\t%d\t%s\t%1.2f\t%1.2f\t%1.2f\t%1.2f\t%d\t%d\t%d\t%1.2f\t%s\n' %
                       (chain, pos, aa, ASAc, ACCc, ASAm, ACCm, inter, intra, Cont, dASA, BCEE))


if __name__ == '__main__':
    # parse_surf_racer()
    # parse_dssp(False)
    # parse_dssp_multichain(False)
    # compute_aledo_classification_multichain()
    compute_aledo_classification()
    # compute_aledo_classification_8ang()
    # compute_aledo_classification_biopython_cox()
    # compute_aledo_classification_cox()