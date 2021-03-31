#!/usr/bin/env python3
import re
import sys 

if len(sys.argv) < 2:
    raise ValueError("Usage: ./1_log2csv.py result.log")

fp = open(sys.argv[1], "r")

dataset_li = []
time_li = []
for line in fp:
    if "Reading from" in line:
        data = re.findall(r'Reading from .*?/.*?\.mtx', line)[0].split('/')[-1].rstrip('.mtx')
        print(data)
        if data not in dataset_li:
            dataset_li.append(data)
    if "elapsed:" in line:
        time = line.split("elapsed:")[1].split("ms")[0]
        print(time)
        time_li.append(time)
fp.close()

fout = open(sys.argv[1].strip(".log")+".csv", 'w')
fout.write("dataset, Avg.Run (ms)\n")
for data, time in zip(dataset_li, time_li):
    fout.write("{},{}\n".format(data, time))

fout.close()
fout.close()