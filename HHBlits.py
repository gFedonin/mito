import subprocess
import os

pathToMSA = '/export/home/fedonin/Epistat/Wuyun/MSA/'
databasePath = '/export/home/fedonin/uniprot20_2016_02/uniprot20_2016_02'
outPath = '/export/home/fedonin/Epistat/HHBlits/'
files = [f for f in os.listdir(pathToMSA)]
processes = set()
max_processes = 144

for name in files:
    command = 'hhblits -cpu 1 -i ' + pathToMSA + name + '/protein_psi.a3m -d ' + databasePath + ' -oa3m ' +\
              outPath + name + '.a3m'
    processes.add(subprocess.Popen(command.split()))
    if len(processes) >= max_processes:
        os.wait()
        processes.difference_update(
            [p for p in processes if p.poll() is not None])
#Check if all the child processes were closed
for p in processes:
    if p.poll() is None:
        p.wait()