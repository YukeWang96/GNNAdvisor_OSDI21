#!/usr/bin/env python3
import os.path as osp
import argparse
import time
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv
from torch.nn import Linear

from dataset import *

run_GCN = False

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='amazon0601', help="dataset")
parser.add_argument("--dim", type=int, default=96, help="input embedding dimension")
parser.add_argument("--hidden", type=int, default=16, help="hidden dimension")
parser.add_argument("--classes", type=int, default=22, help="number of output classes")
parser.add_argument("--epochs", type=int, default=200, help="number of epoches")
args = parser.parse_args()
print(args)

path = osp.join("/home/yuke/.graphs/osdi-ae-graphs", args.dataset+".npz")
dataset = custom_dataset(path, args.dim, args.classes, load_from_txt=False)
data = dataset


if run_GCN:
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = GCNConv(dataset.num_features, args.hidden, cached=True,
                                normalize=False)
            self.conv2 = GCNConv(args.hidden, dataset.num_classes, cached=True,
                                normalize=False)

        def forward(self):
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
            x = F.relu(self.conv1(x, edge_index, edge_weight))
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)
            return F.log_softmax(x, dim=1)
else:
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
            x = self.conv1(x, edge_index)
            x = self.conv2(x, edge_index)
            x = self.conv3(x, edge_index)
            x = self.conv4(x, edge_index)
            x = self.conv5(x, edge_index)
            return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=0.01) 


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
torch.cuda.synchronize()
start = time.perf_counter()
for epoch in tqdm(range(1, args.epochs + 1)):
    train()
    # train_acc, val_acc, tmp_test_acc = test()
    # if val_acc > best_val_acc:
    #     best_val_acc = val_acc
    #     test_acc = tmp_test_acc
    # log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    # print(log.format(epoch, train_acc, best_val_acc, test_acc))
torch.cuda.synchronize()

dur = time.perf_counter() - start
print("GCN (L2-H16) -- Avg Epoch (ms): {:.3f}".format(dur*1e3/args.epochs))
print()