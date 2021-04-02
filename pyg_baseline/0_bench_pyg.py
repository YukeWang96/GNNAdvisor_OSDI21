#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"

# model = 'gcn'
# hidden = [16]

model = 'gin'
hidden = [64]

dataset = [
        ('PROTEINS_full'             , 29       , 2) ,   
        ('OVCAR-8H'                  , 66       , 2) , 
        ('Yeast'                     , 74       , 2) ,
        ('DD'                        , 89       , 2) ,
        ('TWITTER-Real-Graph-Partial', 1323     , 2) ,   
        ('SW-620H'                   , 66       , 2) ,
]


for hid in hidden:
    for data, d, c in dataset:
        command = "python pyg_main.py --dataset {} \
                    --dim {} --hidden {} --classes {}\
                    --model {}".format(data, d, hid, c, model)		
        os.system(command)