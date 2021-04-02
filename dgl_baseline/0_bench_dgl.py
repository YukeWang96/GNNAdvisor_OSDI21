#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"

model = 'gcn'
hidden = [16] 	# for GCN

# model = 'gin'
# hidden = [64] 	# for GIN

dataset = [
		('citeseer'	        , 3703	    , 6   ),  
		('cora' 	        	, 1433	    , 7   ),  
		('pubmed'	        	, 500	    , 3   ),      
		('ppi'	            , 50	    , 121 ),   

		('PROTEINS_full'             , 29       , 2) ,   
		('OVCAR-8H'                  , 66       , 2) , 
		('Yeast'                     , 74       , 2) ,
		('DD'                        , 89       , 2) ,
		('TWITTER-Real-Graph-Partial', 1323     , 2) ,   
		('SW-620H'                   , 66       , 2) ,

		( 'amazon0505'               , 96	  , 22),
		( 'artist'                   , 100    , 12),
		( 'com-amazon'               , 96	  , 22),
		( 'soc-BlogCatalog'	         , 128    , 39), 
		( 'amazon0601'  	         , 96	  , 22), 
]


for hid in hidden:
	for data, d, c in dataset:
		command = "python dgl_main.py --dataset {} \
							--dim {} --hidden {} --classes {} \
							--model {}".format(data, d, hid, c, model)		
		os.system(command)