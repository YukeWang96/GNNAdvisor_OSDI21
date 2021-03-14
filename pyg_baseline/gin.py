#!/usr/bin/env python3
import os.path as osp
import argparse
import time
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GINConv
from torch.nn import Linear

dataset = 'Pubmed' # 'Citeseer' #'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        num_features = dataset.num_features
        dim = 64

        input_fc =  Linear(num_features, dim)
        hidden_fc = Linear(dim, dim)
        output_fc = Linear(dim, dataset.num_classes)

        self.conv1 = GINConv(input_fc)
        self.conv2 = GINConv(hidden_fc)
        self.conv3 = GINConv(hidden_fc)
        self.conv4 = GINConv(hidden_fc)
        self.conv4 = GINConv(hidden_fc)
        self.conv5 = GINConv(output_fc)

    def forward(self):
        x, edge_index = data.x, data.edge_index
        x = self.input_fc(x)
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        x = self.conv5(x, edge_index)
        x = self.output_fc(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()

@torch.no_grad()
def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

# best_val_acc = test_acc = 0
num_epoch = 100
print("=> Profile {} Epoch on {}".format(num_epoch, dataset))

torch.cuda.synchronize()
start = time.perf_counter()
for epoch in tqdm(range(1, num_epoch + 1)):
    train()
    # train_acc, val_acc, tmp_test_acc = test()
    # if val_acc > best_val_acc:
    #     best_val_acc = val_acc
    #     test_acc = tmp_test_acc
    # log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    # print(log.format(epoch, train_acc, best_val_acc, test_acc))
torch.cuda.synchronize()

dur = time.perf_counter() - start
print("GIN (L5-H64) -- Avg Epoch (ms): {:.3f}".format(dur*1e3/num_epoch))