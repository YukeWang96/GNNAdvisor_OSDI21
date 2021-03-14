import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv

class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP"""
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp

    def forward(self, h):
        h = self.mlp(h)
        return h

class GIN(nn.Module):
    """GIN model"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=5):
        """model parameters setting
        Paramters
        ---------
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        learn_eps: boolean
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        """
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.ginlayers = torch.nn.ModuleList()

        num_layers = 5

        for layer in range(self.num_layers):
            if layer == 0:
                mlp = nn.Linear(input_dim, hidden_dim)
            elif layer < self.num_layers - 1:
                mlp = nn.Linear(hidden_dim, hidden_dim) 
            else:
                mlp = nn.Linear(hidden_dim, output_dim) 

            self.ginlayers.append(GINConv(ApplyNodeFunc(nn), "sum", init_eps=0, learn_eps=False))


    def forward(self, g, h):
        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](g, h)
        return h