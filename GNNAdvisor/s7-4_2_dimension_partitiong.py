#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"

dimWorker_li = [1, 2, 4, 8, 16, 32]

dataset = [
		( 'amazon0505'               , 96	, 22),
		( 'artist'                   , 100  , 12),
		( 'com-amazon'               , 96	, 22),
		( 'soc-BlogCatalog'	       	 , 128  , 39), 
		( 'amazon0601'  	         , 96	, 22), 
]

for dimWorker in dimWorker_li:
    print("******************************")
    print("++ dimWorker: {}".format(dimWorker))
    print("******************************")
    for data, d, c in dataset:
        print("{}---dimWorker: {}".format(data, dimWorker))
        print("=================")
        command = "python GNNA_main.py --dataset {} --dim {} --classes {} --dimWorker {}".format(data, d, c, dimWorker)		
        os.system(command)