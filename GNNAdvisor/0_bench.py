#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"

# hidden = [16] 			# for GCN
hidden = [128] 		# for GIN
partsize_li = [30]  	# [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
model = 'gcn'

dataset = [
		# ('tc_gnn_verify'	        , 96	    , 6   ),  
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
			command = "python GNNA_main.py --dataset {} --dim {} --hidden {} --classes {} --partSize {} --model {}".format(data, d, hid, c, partsize, model)		
			os.system(command)