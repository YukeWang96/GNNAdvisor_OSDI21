
import argparse
import time
import os
from tqdm import *
import torch
from dgl.data import register_data_args

from dataset import *

parser = argparse.ArgumentParser()
register_data_args(parser)
parser.add_argument("--dataDir", type=str, default="../osdi-ae-graphs", help="the path to graphs")
parser.add_argument("--gpu", type=int, default=0, help="gpu")
parser.add_argument("--n-epochs", type=int, default=200, help="number of training epochs")
parser.add_argument("--dim", type=int, default=96, help="input embedding dimension")
parser.add_argument("--hidden", type=int, default=16, help="number of hidden gcn units")
parser.add_argument("--classes", type=int, default=10, help="number of output classes")
parser.add_argument("--model", type=str, default='gin', choices=['gcn', 'gin'], help="type of model")
args = parser.parse_args()
print(args)

if args.model == 'gcn':
    from gcn import GCN
else:
    from gin import GIN

def main(args):
    path = os.path.join(args.dataDir, args.dataset+".npz")
    data = custom_dataset(path, args.dim, args.classes, load_from_txt=False)
    g = data.g

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True

    g = g.int().to(args.gpu)

    features = data.x
    labels = data.y
    in_feats = features.size(1)
    n_classes = data.num_classes

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    if args.model == 'gcn':    
        model = GCN(g,
                    in_feats=in_feats,
                    n_hidden=args.hidden,
                    n_classes=n_classes,
                    n_layers=2)
    else:
        model = GIN(g,
                    input_dim=in_feats,
                    hidden_dim=64,
                    output_dim=n_classes,
                    num_layers=5)

    if cuda: model.cuda()

    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-2,
                                 weight_decay=5e-4)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in tqdm(range(args.n_epochs)):
        model.train()

        logits = model(features)
        loss = loss_fcn(logits[:], labels[:])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    dur = time.perf_counter() - start

    if args.model == 'gcn': 
        print("DGL GCN (L2-H16) Time: (ms) {:.3f}". format(dur*1e3/args.n_epochs))
    else:
        print("DGL GIN (L5-H64) Time: (ms) {:.3f}". format(dur*1e3/args.n_epochs))
    print()

if __name__ == '__main__':
    main(args)
