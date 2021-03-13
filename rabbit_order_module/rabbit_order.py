#!/usr/bin/env python3


# import rabbit_reorder as rab
import time
import torch
import sys
import pickle


class graph_input(object):
    def __init__(self, path=None):
        self.load_flag = False
        self.path = path
        
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
                tmp = line.rstrip('\n').split(" ")
                src, dst = int(tmp[0]), int(tmp[1])
                src_li.append(src)
                dst_li.append(dst)
            src_idx = torch.LongTensor(src_li)
            dst_idx = torch.LongTensor(dst_li)
            print(src_idx)
            print(dst_idx)
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
            src_idx = torch.LongTensor(npy_graph[0])
            dst_idx = torch.LongTensor(npy_graph[1])

        dur = time.perf_counter() - start
        print("Loading graph from txt source (ms): {:.3f}".format(dur*1e3))

        self.load_flag = True

    def reorder():
        '''
        reorder the graph if specified.
        '''
        if not self.load_flag: 
            raise ValueError("Graph MUST be loaded Before reordering.")
        
        pass
        

    def create_dgl_graph():
        '''
        create a DGL graph from edge index.
        '''
        if not self.load_flag: 
            raise ValueError("Graph MUST be loaded Before getting PyG graph.")
        
        self.dgl_flag = True
    
    def create_pyg_graph():
        '''
        create a PyG graph from edge index.
        '''
        if not self.load_flag: 
            raise ValueError("Graph MUST be loaded Before getting DGL graph.")
        
        self.pyg_flag = True


    def get_dgl_graph():
        '''
        return the dgl graph.
        '''
        if not self.load_flag: 
            raise ValueError("Graph MUST be loaded Before getting DGL graph.")
        if not self.dgl_flag:
            raise ValueError("DGL Graph MUST be created Before getting DGL graph.")

        return self.dgl_graph

    def get_pyg_graph():
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
    path = "/home/ssd2T_1/yuke/.graphs/orig/cora"
    graph = graph_input(path)
    graph.load()