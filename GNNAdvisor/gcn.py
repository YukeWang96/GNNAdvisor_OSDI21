import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os.path as osp
import argparse
from tqdm import *
import os
import sys
import time
import torch
import math
import numpy as np 
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch.autograd.profiler as profiler

import GNNAdvisor as GNNA
from scipy.sparse import *
from torch_geometric.datasets import Reddit

from gcn_conv import *
from dataset import *

GCN = True
threadPerBlock = 32 # must match the warp-per-block

best_val_acc = test_acc = 0
time_avg = []
test_time_avg = []

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='amazon0601', help="dataset")
parser.add_argument("--dim", type=int, default=96, help="input embedding dimension")
parser.add_argument("--hidden", type=int, default=16, help="hidden dimension")
parser.add_argument("--classes", type=int, default=22, help="number of output classes")
parser.add_argument("--partsize", type=int, default=22, help="neighbor-group size")
args = parser.parse_args()

partsize = args.partsize # 512
dataset = args.dataset

path = osp.join("/home/yuke/.graphs/osdi-ae-graphs/", dataset+".npz")
print(path)
data = custom_dataset(path, args.dim, args.classes, load_from_txt=False)
dataset = data

num_nodes = len(data.x)
num_edges = len(data.edge_index[1])
val = [1]*num_edges

start = time.perf_counter()
scipy_coo = coo_matrix((val, data.edge_index), shape=(num_nodes,num_nodes))
scipy_csr = scipy_coo.tocsr()
build_csr = time.perf_counter() - start

print("Build CSR (s): {:.3f}".format(build_csr))
column_index = torch.IntTensor(scipy_csr.indices)
row_pointers = torch.IntTensor(scipy_csr.indptr)

def func(x):
    if x > 0:
        return x
    else:
        return 1

degrees = (row_pointers[1:] - row_pointers[:-1]).tolist()
degrees = torch.sqrt(torch.FloatTensor(list(map(func, degrees)))).cuda()

start = time.perf_counter()
partPtr, part2Node = GNNA.build_part(partsize, row_pointers)
build_neighbor_parts = time.perf_counter() - start
print("Build nb_part (s): {:.3f}".format(build_neighbor_parts))

# partPtr, part2Node = part_based_partitioing(scipy_csr.indptr, scipy_csr.indices)
# partPtr = torch.IntTensor(partPtr).cuda()
# part2Node = torch.IntTensor(part2Node).cuda()
partPtr = partPtr.int().cuda()
part2Node = part2Node.int().cuda()
# print(partPtr)
# print(part2Node)
column_index = column_index.cuda()
row_pointers = row_pointers.cuda()
inputInfo = inputProperty(row_pointers, column_index, degrees, 
                            partPtr, part2Node, threadPerBlock)

if GCN:
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = GCNConv(dataset.num_features, 16)
            self.conv2 = GCNConv(16, dataset.num_classes)

        def forward(self):
            x = data.x
            x = F.relu(self.conv1(x, inputInfo))
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, inputInfo)
            return F.log_softmax(x, dim=1)
else:
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = GINConv(dataset.num_features, args.hidden)
            self.conv2 = GINConv(args.hidden, args.hidden)
            self.conv3 = GINConv(args.hidden, args.hidden)
            self.conv4 = GINConv(args.hidden, args.hidden)
            self.conv5 = GINConv(args.hidden, dataset.num_classes)

        def forward(self):
            x = data.x
            x = F.relu(self.conv1(x, inputInfo))
            x = F.relu(self.conv2(x, inputInfo))
            x = F.relu(self.conv3(x, inputInfo))
            x = F.relu(self.conv4(x, inputInfo))
            x = self.conv5(x, inputInfo)
            return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    loss = F.nll_loss(model()[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

@torch.no_grad()
def test(profile=False):
    model.eval()
    if profile:
        with profiler.profile(record_shapes=True, use_cuda=True) as prof:
            with profiler.record_function("model_inference"):
                logits, accs = model(), []
        print(prof.key_averages().table())
    else:
        logits, accs = model(), []
    
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


num_epoches = 200
for epoch in tqdm(range(1, num_epoches + 1)):
    start_train = time.perf_counter()
    train()
    train_time = time.perf_counter() - start_train
    time_avg.append(train_time)
    # if epoch == 10:
    #     # break
    #     train_acc, val_acc, tmp_test_acc = test(profile=True)
    #     # break
    # else:
    # start_test = time.perf_counter()
    # train_acc, val_acc, tmp_test_acc = test()
    # test_time = time.perf_counter() - start_test
    # test_time_avg.append(test_time)
    
    # if val_acc > best_val_acc:
    #     best_val_acc = val_acc
    #     test_acc = tmp_test_acc
    # log = 'Epoch: {:03d}, Train: {:.4f}, Train-Time: {:.3f} ms, Test-Time: {:.3f} ms, Val: {:.4f}, Test: {:.4f}'
    # print(log.format(epoch, train_acc, sum(time_avg)/len(time_avg) * 1e3, sum(test_time_avg)/len(test_time_avg) * 1e3, best_val_acc, test_acc))

print('Avg. Train-Time (ms): {:.3f}'.format(np.mean(time_avg)*1e3))