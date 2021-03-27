#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"

partsize_li = [2, 4, 8, 16, 32, 64, 128, 256, 512]

dataset = [
        ( 'amazon0505'               , 96	, 22),
        ( 'artist'                   , 100  , 12),
        ( 'com-amazon'               , 96	, 22),
        ( 'soc-BlogCatalog'	       	 , 128  , 39), 
        ( 'amazon0601'  	         , 96	, 22), 
]

for partsize in partsize_li:
    print("******************************")
    print("++ Part-size: {}".format(partsize))
    print("******************************")
    for data, d, c in dataset:
        print("{}---partsize: {}".format(data, partsize))
        print("=================")
        command = "python GNNA_main.py --dataset {} --dim {} --classes {} --partSize {}".format(data, d, c, partsize)		
        os.system(command)