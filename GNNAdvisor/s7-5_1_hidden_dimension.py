#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"

hidden_li = [16, 32, 64, 128, 256, 512, 1024] 

dataset = [
        ( 'amazon0505'               , 96	, 22),
        ( 'artist'                   , 100  , 12),
        ( 'com-amazon'               , 96	, 22),
        ( 'soc-BlogCatalog'	       	 , 128  , 39), 
        ( 'amazon0601'  	         , 96	, 22), 
]

for hidden in hidden_li:
    print("******************************")
    print("++ hiddenDimension: {}".format(hidden))
    print("******************************")
    for data, d, c in dataset:
        print("{}---hidden: {}".format(data, hidden))
        print("=================")
        command = "python GNNA_main.py --dataset {} --dim {} --hidden {} --classes {}".format(data, d, hidden, c)		
        os.system(command)