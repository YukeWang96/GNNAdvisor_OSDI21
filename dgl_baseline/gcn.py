import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()

        assert n_layers >= 2
        self.layers.append(GraphConv(in_feats, n_hidden, allow_zero_in_degree=True))

        for i in range(n_layers - 2):
            self.layers.append(GraphConv(n_hidden, n_hidden, allow_zero_in_degree=True))
        
        self.layers.append(GraphConv(n_hidden, n_classes, allow_zero_in_degree=True))

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(self.g, h)
        return F.log_softmax(h, dim=1)
