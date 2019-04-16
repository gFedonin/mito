path_to_pdb = '../pdb/5ara.pdb1'
out_path = '../pdb/5ara_W.pdb1'
chain = 'W'

if __name__ == '__main__':
    with open(out_path, 'w') as f:
        for line in open(path_to_pdb, 'r').readlines():
            s = line.strip().split()
            if s[0] == 'ATOM' and s[4] == chain:
                f.write(line)