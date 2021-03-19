#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"

hidden = [16] 		# [64] 		# [16, 32, 64, 128, 256, 512, 1024, 2048]
partsize_li = [32]  # [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:

dataset = [
		('citeseer'	        , 3703	    , 6   ),  
		('cora' 	        , 1433	    , 7   ),  
		('pubmed'	        , 500	    , 3   ),      
		('ppi'	            , 50	    , 121 ),   

		# ('PROTEINS_full'             , 29       , 2) ,   
		# ('OVCAR-8H'                  , 66       , 2) , 
		# ('Yeast'                     , 74       , 2) ,
		# ('DD'                        , 89       , 2) ,
		# ('TWITTER-Real-Graph-Partial', 1323     , 2) ,   
		# ('SW-620H'                   , 66       , 2) ,
		
		# ( 'amazon0505'               , 96	, 22),
		# ( 'artist'                   , 100  , 12),
		# ( 'com-amazon'               , 96	, 22),
		# ( 'soc-BlogCatalog'	       	 , 128  , 39), 
		# ( 'amazon0601'  	         , 96	, 22), 
]


for partsize in partsize_li:
	for hid in hidden:
		for data, d, c in dataset:
			command = "python gnn.py --dataset {} --dim {} --hidden {} --classes {} --partsize {}".format(data, d, hid, c, partsize)		
			os.system(command)
			print()