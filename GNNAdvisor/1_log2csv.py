#!/usr/bin/env python3
import re
import sys 

if len(sys.argv) < 2:
    raise ValueError("Usage: ./1_log2csv.py result.log")

fp = open(sys.argv[1], "r")

dataset_li = []
time_li = []
for line in fp:
    if "dataset=" in line:
        data = re.findall(r'dataset=.*?,', line)[0].split('=')[1].replace(",", "").replace('\'', "")
        print(data)
        dataset_li.append(data)
    if "Time (ms):" in line:
        time = line.split("Time (ms):")[1].rstrip("\n")
        print(time)
        time_li.append(time)
fp.close()

fout = open(sys.argv[1].strip(".log")+".csv", 'w')
fout.write("dataset,Avg.Epoch (ms)\n")
for data, time in zip(dataset_li, time_li):
    fout.write("{},{}\n".format(data, time))

fout.close()
fout.close()