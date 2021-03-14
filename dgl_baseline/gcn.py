import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
        
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
        self.layers.append(GraphConv(in_feats, n_hidden))

        for i in range(n_layers - 2):
            self.layers.append(GraphConv(n_hidden, n_hidden))
        
        self.layers.append(GraphConv(n_hidden, n_classes))

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(self.g, h)
        return h
