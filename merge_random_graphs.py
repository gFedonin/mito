from os import listdir
from os.path import exists

path_to_main_set = '../res/random_graph_stat_hist_cytb_Aledo_igraph'
# additional_sets = range(5, 10)
out_path = '../res/cytb_Aledo_igraph_merged.random_graphs'
prot_name = 'cytb/'


def parse_dir(dir_path):
    res = []
    if exists(dir_path):
        for file in listdir(dir_path):
            res.extend(open(dir_path + file, 'r').readlines())
    return res


if __name__ == '__main__':
    graphs = parse_dir(path_to_main_set + '/temp/' + prot_name)
    # for i in additional_sets:
    #     graphs.extend(parse_dir(path_to_main_set + str(i) + '/temp/' + prot_name))
    print('total graphs = ' + str(len(graphs)))
    with open(out_path, 'w') as out:
        out.write(''.join(graphs))
