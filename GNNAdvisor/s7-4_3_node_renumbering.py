#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"

node_renumbering_li = ['True', 'False']

dataset = [
		( 'amazon0505'               , 96	, 22),
		( 'artist'                   , 100  , 12),
		( 'com-amazon'               , 96	, 22),
		( 'soc-BlogCatalog'	       	 , 128  , 39), 
		( 'amazon0601'  	         , 96	, 22), 
]

for nodeRenumber in node_renumbering_li:
    print("******************************")
    print("++ NodeRenumber: {}".format(node_renumbering))
    print("******************************")
    for data, d, c in dataset:
        print("{}---NodeRenumber: {}".format(data, nodeRenumber))
        print("=================")
        command = "python GNNA_main.py --dataset {} --dim {} --classes {} --enable_rabbit {}".format(data, d, c, dimWorker, nodeRenumber)		
        os.system(command)