#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"

hidden = [16]
dataset = [
        ( 'amazon0505'               , 96	  , 22),
        ( 'artist'                   , 100  , 12),
        ( 'com-amazon'               , 96	  , 22),
        ( 'soc-BlogCatalog'	         , 128  , 39), 
        ( 'amazon0601'  	         , 96	  , 22), 
]


for hid in hidden:
    for data, _, _ in dataset:
        command = "./spmm market ../osdi-ae-graphs-mtx/{}.mtx --num-runs=200 --feature-len={}".format(data, hid)		
        os.system(command)