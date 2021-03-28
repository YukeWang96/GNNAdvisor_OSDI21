#!/usr/bin/env python3
import time
import torch

import pickle
import rabbit

class graph_input(object):
    def __init__(self, path=None):
        self.load_flag = False
        self.reorder_flag = False
        self.path = path
        self.edge_index = None
        
        self.dgl_flag = False
        self.pyg_flag = False

        self.dgl_graph = False
        self.pyg_graph = False


    def load(self, load_from_txt=True):
        '''
        load the graph from the disk --> CPU memory.
        '''
        if self.path == None:
            raise ValueError("Graph path must be assigned first")
        
        start = time.perf_counter()
        if load_from_txt:
            '''
            edge in the txt format:
            s0 d0
            s1 d1
            s2 d2
            '''
            fp = open(self.path, "r")
            src_li = []
            dst_li = []
            for line in fp:
                tmp = line.rstrip('\n').split()
                src, dst = int(tmp[0]), int(tmp[1])
                src_li.append(src)
                dst_li.append(dst)
            src_idx = torch.IntTensor(src_li)
            dst_idx = torch.IntTensor(dst_li)
            self.edge_index = torch.stack([src_idx, dst_idx], dim=0)
            # print(self.edge_index)
            # print(src_idx)
            # print(dst_idx)
        else:
            '''
            graph must store in a numpy object with the shape of [2, num_edges].
            [
                [s0, s1, s2, ... , sn],
                [d0, d1, d2, ... , dn],
            ]
            expected loading speed is faster than loading from txt.
            '''
            fp = open(self.path, "rb")
            npy_graph = pickle.load(fp)
            src_idx = torch.IntTensor(npy_graph[0])
            dst_idx = torch.IntTensor(npy_graph[1])

        dur = time.perf_counter() - start
        print("Loading graph from txt source (ms): {:.3f}".format(dur*1e3))

        self.load_flag = True

    def reorder(self):
        '''
        reorder the graph if specified.
        '''
        if not self.load_flag: 
            raise ValueError("Graph MUST be loaded Before reordering.")
        
        print("Original edge_index\n", self.edge_index)
        new_edge_index = rabbit.reorder(self.edge_index)
        print("Reordered edge_index\n", new_edge_index)

        # for i in range(len(new_edge_index[1])):
        #     src, dst = new_edge_index[0][i], new_edge_index[1][i]
            # print('{}--{}'.format(src, dst))
        # print(new_edge_index.size())

        self.reorder_flag = True
        

    def create_dgl_graph(self):
        '''
        create a DGL graph from edge index.
        '''
        if not self.load_flag: 
            raise ValueError("Graph MUST be loaded Before getting PyG graph.")
        
        self.dgl_flag = True
    
    def create_pyg_graph(self):
        '''
        create a PyG graph from edge index.
        '''
        if not self.load_flag: 
            raise ValueError("Graph MUST be loaded Before getting DGL graph.")
        
        self.pyg_flag = True


    def get_dgl_graph(self):
        '''
        return the dgl graph.
        '''
        if not self.load_flag: 
            raise ValueError("Graph MUST be loaded Before getting DGL graph.")
        if not self.dgl_flag:
            raise ValueError("DGL Graph MUST be created Before getting DGL graph.")

        return self.dgl_graph

    def get_pyg_graph(self):
        '''
        return the pyg graph.
        '''
        if not self.load_flag: 
            raise ValueError("Graph MUST be loaded Before getting PyG graph.")

        if not self.pyg_flag:
            raise ValueError("PyG Graph MUST be created Before getting PyG graph.")
        
        return self.pyg_graph


if __name__ == "__main__":
    # path = "/home/ssd2T_1/yuke/.graphs/orig/amazon0505"
    # path = "/home/ssd2T_1/yuke/.graphs/orig/cora"
    path = "/home/yuke/.graphs/orig/toy"
    graph = graph_input(path)
    graph.load(load_from_txt=True)
    graph.reorder()