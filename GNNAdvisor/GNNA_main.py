import os
import sys
import time
import argparse
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F
from tqdm import *
from scipy.sparse import *

import GNNAdvisor as GNNA           # import GNNAdvisor

from gnn_conv import *
from dataset import *

# Verify single sparse kernel
TEST = False    
if TEST == True:
    from unitest import *

parser = argparse.ArgumentParser()
# Dataset related parameters.
parser.add_argument("--dataDir", type=str, default="../osdi-ae-graphs", help="the path to graphs")
parser.add_argument("--dataset", type=str, default='amazon0601', help="dataset")
parser.add_argument("--dim", type=int, default=96, help="input embedding dimension size")
parser.add_argument("--hidden", type=int, default=16, help="hidden dimension size")
parser.add_argument("--classes", type=int, default=22, help="output classes size")

# Manually set the performance related parameters
parser.add_argument("--partSize", type=int, default=32, help="neighbor-group size")
parser.add_argument("--dimWorker", type=int, default=32, help="number of worker threads (MUST < 32)")
parser.add_argument("--warpPerBlock", type=int, default=2, help="number of warp per block, recommended: GCN: 8, GIN: 2")
parser.add_argument("--sharedMem", type=int, default=100, help="shared memory size of each block (Quadro P6000 64KB), default=100 KB for RTX3090")

parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gin'],  help="GCN or GIN")
parser.add_argument("--num_epoches", type=int, default=200, help="number of epoches for training, default=200")

parser.add_argument('-loadFromTxt', action='store_true', help="whether to load the graph TXT edge list, default: False (load from npz fast)")
parser.add_argument('-enable_rabbit', action='store_true', help="whether to enable rabbit reordering, default: False for both manual and auto mode.")
parser.add_argument('-manual_mode', action='store_true', help="whether to use manual config, defuatl: auto config mode")
parser.add_argument('-verbose_mode', action='store_true', help="whether to use manual config, defuatl: auto config mode")

args = parser.parse_args()
if args.verbose_mode:
    print(args)
partSize, dimWorker, warpPerBlock, sharedMem = args.partSize, args.dimWorker, args.warpPerBlock, args.sharedMem

# requires GPU for evaluation.
assert torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loading data from files
if args.loadFromTxt:
    path = osp.join(args.dataDir, args.dataset)
    dataset = custom_dataset(path, args.dim, args.classes, load_from_txt=True, verbose=args.verbose_mode)
    # path = osp.join("/home/yuke/.graphs/orig_rabbit", dataset)
else:
    path = osp.join(args.dataDir, args.dataset+".npz")
    dataset = custom_dataset(path, args.dim, args.classes, load_from_txt=False, verbose=args.verbose_mode)

num_nodes = dataset.num_nodes
num_edges = dataset.num_edges
column_index = dataset.column_index
row_pointers = dataset.row_pointers
degrees = dataset.degrees

# Building neighbor partitioning.
start = time.perf_counter()
partPtr, part2Node = GNNA.build_part(partSize, row_pointers)
build_neighbor_parts = time.perf_counter() - start
if args.verbose_mode:
    print("# Build nb_part (s): {:.3f}".format(build_neighbor_parts))

partPtr = partPtr.int().to(device)
part2Node = part2Node.int().to(device)
column_index = column_index.to(device)
row_pointers = row_pointers.to(device)

# dimWorker = 10
# Building input property profile.
inputInfo = inputProperty(row_pointers, column_index, degrees, 
                            partPtr, part2Node,
                            partSize, dimWorker, warpPerBlock, sharedMem,
                            hiddenDim=args.hidden, dataset_obj=dataset, enable_rabbit=args.enable_rabbit,
                            manual_mode=True, verbose=args.verbose_mode)

inputInfo.decider()

inputInfo = inputInfo.set_input()
if args.verbose_mode:
    print('----------------------------')
    inputInfo.print_param()
    print()

inputInfo = inputInfo.set_hidden()
if args.verbose_mode:
    inputInfo.print_param()
    print()
    print('----------------------------')
# sys.exit(0)

if TEST:
    valid = Verification(row_pointers, column_index, degrees, partPtr, part2Node, \
                        partSize, dimWorker, warpPerBlock)
    valid.compute()
    # valid.reference()
    # valid.compare()
    sys.exit(0)

# Building GNN model
if args.model == 'gcn':
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = GCNConv(dataset.num_features, args.hidden)
            self.conv2 = GCNConv(args.hidden, dataset.num_classes)

        def forward(self):
            x = dataset.x
            x = F.relu(self.conv1(x, inputInfo.set_input()))
            x = self.conv2(x, inputInfo.set_hidden())
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
            x = dataset.x
            x = F.relu(self.conv1(x, inputInfo.set_input()))
            x = F.relu(self.conv2(x, inputInfo.set_hidden()))
            x = F.relu(self.conv3(x, inputInfo.set_hidden()))
            x = F.relu(self.conv4(x, inputInfo.set_hidden()))
            x = self.conv5(x, inputInfo.set_hidden())
            return F.log_softmax(x, dim=1)

model, dataset = Net().to(device), dataset.to(device)
if args.verbose_mode:
    print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Define training function.
def train():
    model.train()
    optimizer.zero_grad()
    loss = F.nll_loss(model()[dataset.train_mask], dataset.y[dataset.train_mask])
    loss.backward()
    optimizer.step()

# Training iteration begin.
time_avg = []
for epoch in tqdm(range(1, args.num_epoches + 1)):
    start_train = time.perf_counter()
    train()
    train_time = time.perf_counter() - start_train
    time_avg.append(train_time)

print('GNNAdvisor Time (ms): {:.3f}'.format(np.mean(time_avg)*1e3))
print()