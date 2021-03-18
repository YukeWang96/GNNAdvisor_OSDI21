
import argparse
import time
import os
import numpy as np
from tqdm import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import register_data_args

from dataset import *

run_GCN = False
if run_GCN:
    from gcn import GCN
else:
    from gin import GIN

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def main(args):
    path = os.path.join("/home/yuke/.graphs/osdi-ae-graphs", args.dataset+".npz")
    data = custom_dataset(path, args.dim, args.classes, load_from_txt=False)
    g = data.g

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True

    g = g.int().to(args.gpu)

    features = data.x
    labels = data.y
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    in_feats = features.size(1)
    n_classes = data.num_classes
    n_edges = data.num_edges

    # add self loop
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    if run_GCN:    
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
    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in tqdm(range(args.n_epochs)):
        model.train()
        # if epoch >= 3:
        t0 = time.time()

        # forward
        logits = model(features)
        loss = loss_fcn(logits[:], labels[:])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if epoch >= 3:
        dur.append(time.time() - t0)
        # acc = evaluate(model, features, labels, val_mask)
    
    print("DGL Time: (ms) {:.3f}". format(np.mean(dur)*1e3))
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")

    parser.add_argument("--dim", type=int, default=96, 
                        help="input embedding dimension")
    parser.add_argument("--hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--classes", type=int, default=10,
                        help="number of output classes")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")

    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)
