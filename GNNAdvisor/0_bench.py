#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"

run_GCN = True
manual_mode = True
verbose_mode = False

if run_GCN:
    model = 'gcn'
    partSize = 8
    dimWorker = 32
    warpPerBlock = 8
    hidden = [16] 
else:
    model = 'gin'
    partSize = 8
    dimWorker = 32
    warpPerBlock = 6
    hidden = [64] 		

partsize_li = [32]

dataset = [
        # ('toy'	        , 3	    , 2   ),  
        # ('tc_gnn_verify'	, 16	, 2),
        # ('tc_gnn_verify_2x'	, 16	, 2),

        ('citeseer'	        , 3703	    , 6   ),  
        ('cora' 	        , 1433	    , 7   ),  
        ('pubmed'	        , 500	    , 3   ),      
        ('ppi'	            , 50	    , 121 ),   

        ('PROTEINS_full'             , 29       , 2) ,   
        ('OVCAR-8H'                  , 66       , 2) , 
        ('Yeast'                     , 74       , 2) ,
        ('DD'                        , 89       , 2) ,
        ('TWITTER-Real-Graph-Partial', 1323     , 2) ,   
        ('SW-620H'                   , 66       , 2) ,
        
        ( 'amazon0505'               , 96	, 22),
        ( 'artist'                   , 100  , 12),
        ( 'com-amazon'               , 96	, 22),
        ( 'soc-BlogCatalog'	       	 , 128  , 39), 
        ( 'amazon0601'  	         , 96	, 22), 
]


for partsize in partsize_li:
    for hid in hidden:
        for data, d, c in dataset:
            command = "python GNNA_main.py --dataset {} --dim {} --hidden {} \
                        --classes {} --model {} \
                        --partSize {} \
                        --dimWorker {} \
                        --warpPerBlock {} \
                        --manual_mode {} \
                        --verbose_mode {}"
            command = command.format(data, d, hid, c, model, partsize, dimWorker, \
                                        warpPerBlock, manual_mode, verbose_mode)		
            # command = "python GNNA_main.py -loadFromTxt --dataset {} --partSize {} --dataDir {}".format(data, partsize, '/home/yuke/.graphs/orig')		 
            os.system(command)
