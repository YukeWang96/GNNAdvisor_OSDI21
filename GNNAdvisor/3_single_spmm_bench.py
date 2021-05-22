#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"

manual_mode = True
verbose_mode = False
hidden = 16 

dataset = [
		( 'amazon0505'               , 96	, 22),
		( 'artist'                   , 100  , 12),
		( 'com-amazon'               , 96	, 22),
		( 'soc-BlogCatalog'	       	 , 128  , 39), 
		( 'amazon0601'  	         , 96	, 22), 
]

print("******************************")
print("++ Single SpMM Kernel -- Dim: {}".format(hidden))
print("******************************")
for data, _, _ in dataset:
    command = "python GNNA_main.py  \
            --dataset {} \
            --hidden {} \
            --verbose_mode {} \
            --manual_mode {} \
            --single_spmm {}".format(data, \
                                    hidden, \
                                    verbose_mode, \
                                    manual_mode, \
                                    True)		
    os.system(command)
    print("=================")
