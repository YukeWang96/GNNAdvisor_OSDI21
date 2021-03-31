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