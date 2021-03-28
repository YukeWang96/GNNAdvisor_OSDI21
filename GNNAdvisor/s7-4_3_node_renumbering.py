#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"

node_renumbering_li = ['False', 'True']
manual_mode = True
verbose_mode = False

run_GCN = False
if run_GCN:
    model = 'gcn'
    # warpPerBlock = 8
    hidden = 16 
else:
    model = 'gin'
    # warpPerBlock = 6
    hidden = 64 	

dataset = [
		( 'amazon0505'               , 96	, 22),
		( 'artist'                   , 100  , 12),
		( 'com-amazon'               , 96	, 22),
		# ( 'soc-BlogCatalog'	       	 , 128  , 39), 
		# ( 'amazon0601'  	         , 96	, 22), 
]

for nodeRenumber in node_renumbering_li:
    print("******************************")
    print("++ NodeRenumber: {}".format(nodeRenumber))
    print("******************************")
    for data, d, c in dataset:
        print("{}---NodeRenumber: {}".format(data, nodeRenumber))
        print("=================")
        command = "python GNNA_main.py  \
                --dataset {} \
                --dim {} \
                --hidden {} \
                --classes {} \
                --enable_rabbit {} \
                --verbose_mode {} \
                --manual_mode {} \
                --model {}".format(data, d, hidden, c, nodeRenumber, \
                                        verbose_mode, manual_mode, model)		
        os.system(command)