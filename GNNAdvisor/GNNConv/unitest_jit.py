#!/usr/bin/env python3
import torch
import torch.nn as nn
import GNNAdvisor as GNNA

X = torch.rand((6,4,1,1)).cuda()
w = torch.rand((7, 2)).cuda()
out = torch.zeros((6,7,1,1)).cuda()
overlap = 0.5
output = GNNA.forward(X, w, out, overlap)
print(output[0])
print(output[0].size())

d_out = torch.rand((6,7,1,1)).cuda()
X = torch.randn((6,4,1,1)).cuda()
w = torch.randn((7,2)).cuda()
d_X = torch.zeros_like(X).cuda()
d_w = torch.zeros_like(w).cuda()
overlap = 0.5
d_output = GNNA.backward(d_out, X, w, d_X, d_w, overlap)
print("d_X: ", d_output[0])
print("d_w: ", d_output[1])