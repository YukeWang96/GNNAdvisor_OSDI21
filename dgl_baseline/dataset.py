#!/usr/bin/env python3
import torch
import numpy as np
import time
import dgl 
import sys

class custom_dataset(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, path, dim, num_class, load_from_txt=True):
        super(custom_dataset, self).__init__()
        self.nodes = set()

        self.load_from_txt = load_from_txt
        self.num_nodes = 0
        self.num_features = dim 
        self.num_classes = num_class
        
        self.init_edges(path)
        self.init_embedding(dim)
        self.init_labels(num_class)

        train = 1
        val = 0.3
        test = 0.1
        self.train_mask = [1] * int(self.num_nodes * train) + [0] * (self.num_nodes  - int(self.num_nodes * train))
        self.val_mask = [1] * int(self.num_nodes * val)+ [0] * (self.num_nodes  - int(self.num_nodes * val))
        self.test_mask = [1] * int(self.num_nodes * test) + [0] * (self.num_nodes  - int(self.num_nodes * test))
        self.train_mask = torch.BoolTensor(self.train_mask).cuda()
        self.val_mask = torch.BoolTensor(self.val_mask).cuda()
        self.test_mask = torch.BoolTensor(self.test_mask).cuda()

    def init_edges(self, path):
        self.g = dgl.DGLGraph()

        # loading from a txt graph file
        if self.load_from_txt:
            fp = open(path, "r")
            src_li = []
            dst_li = []
            start = time.perf_counter()
            for line in fp:
                src, dst = line.strip('\n').split()
                src, dst = int(src), int(dst)
                src_li.append(src)
                dst_li.append(dst)
                self.nodes.add(src)
                self.nodes.add(dst)
            self.g.add_edges(src_li, dst_li)
            self.num_edges = len(src_li)
            self.num_nodes = max(self.nodes) + 1
            dur = time.perf_counter() - start
            print("# Loading (txt) {:.3f}s ".format(dur))

        # loading from a .npz graph file
        else: 
            if not path.endswith('.npz'):
                raise ValueError("graph file must be a .npz file")

            start = time.perf_counter()
            
            graph_obj = np.load(path)
            src_li = graph_obj['src_li']
            dst_li = graph_obj['dst_li']
            self.num_nodes = graph_obj['num_nodes']
            self.g.add_edges(src_li, dst_li)
            self.num_edges = len(src_li)

            dur = time.perf_counter() - start
            print("# Loading (npz): {:.3f}s ".format(dur))

    def init_embedding(self, dim):
        '''
        Generate node embedding for nodes.
        '''
        self.x = torch.randn(self.num_nodes, dim).cuda()
    
    def init_labels(self, num_class):
        '''
        Generate the node label.
        '''
        self.y = torch.ones(self.num_nodes).long().cuda()

    def forward(*input, **kwargs):
        pass
