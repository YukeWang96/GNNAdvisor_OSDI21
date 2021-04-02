#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"

run_GCN = True
enable_rabbit = True
manual_mode = False
verbose_mode = True
loadFromTxt = False

if run_GCN:
    model = 'gcn'
    warpPerBlock = 8
    hidden = [16] 
else:
    model = 'gin'
    warpPerBlock = 6
    hidden = [64] 		

partsize_li = [2]

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
                        --classes {} --partSize {} --model {} --warpPerBlock {}\
                        --manual_mode {} --verbose_mode {} --enable_rabbit {} --loadFromTxt {}"
            command = command.format(data, d, hid, c, partsize, model, warpPerBlock,\
                                     manual_mode, verbose_mode, enable_rabbit, loadFromTxt)		
            # command = "python GNNA_main.py -loadFromTxt --dataset {} --partSize {} --dataDir {}".format(data, partsize, '/home/yuke/.graphs/orig')		 
            os.system(command)
