#!/usr/bin/env python3
import sys 

if len(sys.argv) < 2:
    raise ValueError("Usage: ./1_log2csv.py result.log")

fp = open(sys.argv[1], "r")

param_li = []
dataset_li = []
time_li = []
for line in fp:
    if  "++" in line:
        param_li.append(line.split(":")[1])
    if "---" in line:
        data = line.split('---')[0]
        # print(data)
        if data not in dataset_li:
            dataset_li.append(data)
    if "Time (ms):" in line:
        time = line.split("Time (ms):")[1].rstrip("\n")
        # print(time)
        time_li.append(time)
fp.close()

fout = open(sys.argv[1].strip(".log")+".csv", 'w')
line = "dataset"
for param in param_li:
    line += "," + param.strip('\n')
fout.write(line+'\n')
print(line)
# print(dataset_li)
# print(time_li)
for i, data in enumerate(dataset_li):
    line = data
    for t_idx in range(i, len(time_li), len(dataset_li)):
        line += ",{}".format(time_li[t_idx].rstrip('\n'))
    fout.write(line+"\n")
    print(line)
fout.close()