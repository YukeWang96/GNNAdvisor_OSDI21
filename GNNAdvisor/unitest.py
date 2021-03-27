#!/usr/bin/env python3
import torch
import GNNAdvisor as GNNA
import sys

class Verification(object):
    def __init__(self, row_pointers, column_index, degrees, partPtr, part2Node, \
                partSize, dimWorker, warpPerBlock):

        self.row_pointers = row_pointers
        self.column_index = column_index
        self.degrees = degrees
        self.partPtr = partPtr
        self.part2Node = part2Node

        self.warpPerBlock = warpPerBlock      
        self.partSize = partSize
        self.dimWorker = dimWorker

        self.num_nodes = len(row_pointers) - 1
        self.test_embedding = 3
        self.output_embedding = 3

        self.X = torch.ones(self.num_nodes, self.test_embedding)
        self.W = torch.ones(self.test_embedding, self.output_embedding)

        self.result = None
        self.result_ref = None
        
    def reference(self):
        print("# Compute reference on CPU")
        tmp = torch.mm(self.X, self.W)
        self.result_ref = torch.zeros_like(tmp)

        for i in range(len(self.row_pointers) - 1):
            for eidx in range(self.row_pointers[i], self.row_pointers[i+1]):
                for d in range(len(tmp[0])):
                    eid = self.column_index[eidx]
                    self.result_ref[i][d] += tmp[eid][d]
        # print(self.result_ref)

    def compute(self):
        print("# Compute on GPU")
        X = self.X.cuda()
        # W = self.W.cuda()
        # print(X.size())
        # print(self.row_pointers)
        # print(self.column_index)
        # print(self.partPtr)
        # print(self.part2Node)
        # self.result = GNNA.forward(X, W, self.row_pointers, self.column_index, self.degrees,\
                                    # self.partPtr, self.part2Node, self.partSize, self.dimWorker, self.warpPerBlock)[0]
        
        self.result = GNNA.SAG(X, self.row_pointers, self.column_index, self.degrees,\
                                    self.partPtr, self.part2Node, self.partSize, self.dimWorker, self.warpPerBlock)
        # print(self.result)
        print("finished")


    def compare(self):
        if self.result_ref is None or self.result is None:
            raise ValueError("MUST compute result and result reference first!!")
        assert torch.all(torch.eq(self.result_ref, self.result.cpu()))
        print("PASS")