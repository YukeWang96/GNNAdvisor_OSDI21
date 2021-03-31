#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"

manual_mode = True
verbose_mode = False
hidden = 16 

dataset = [
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

print("******************************")
print("++ Verifying SpMM Kernel -- Dim: {}".format(hidden))
print("******************************")
for data, _, _ in dataset:
    print("=================")
    command = "python GNNA_main.py  \
            --dataset {} \
            --hidden {} \
            --verbose_mode {} \
            --manual_mode {} \
            --verify_spmm {}".format(data, \
                                    hidden, \
                                    verbose_mode, \
                                    manual_mode, \
                                    True)		
    os.system(command)