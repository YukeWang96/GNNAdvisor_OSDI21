# GNNAdvisor 

## Dependency 
+ **`libboost`** 
> `sudo apt-get install libboost-all-dev`
+ **`tcmalloc`**
> `sudo apt-get install libgoogle-perftools-dev`

+ conda
```
conda create -n env_name python=3.6
```
**Note that make sure the `pip` you use is the `pip` installed inside the current conda environment. You can check this by `which pip`**

+ Install `Pytorch`
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
```
or 
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```
> 
+ Install [`Deep Graph Library (DGL)`](https://github.com/dmlc/dgl).
```
conda install -c dglteam dgl-cuda11.0
pip install torch requests
```

+ Install [`Pytorch-Geometric (PyG)`](https://github.com/rusty1s/pytorch_geometric).
```
CUDA=cu111
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-geometric
```

## Reproduce the experiment in paper.
+ **GNN Model Setting**.
> + **GCN (2-layer with 16 hidden dimension)**
> + **GIN (5-layer with 64 hidden dimension)**
+ **Datasets**.
> **Type I**:
> `citeseer, cora, pubmed, ppi`

> **Type II**:
> `PROTEINS_full, OVCAR-8H, Yeast, DD, TWITTER-Real-Graph-Partial, SW-620H`

> **Type III**:
>`amazon0505, artist, com-amazon, soc-BlogCatalog, amazon0601`

+ Running **DGL** baseline on GNN training.
> +  Go to **`dgl_baseline/`** directory
> + `run_GCN` in `train.py` set to `True` to profiling the GCN model;
> + `run_GCN` in `train.py` set to `False` to profiling the GIN model; 
> + `./bench` to run the script and the report 200 epoch runtime for all evaluated datasets. 

+ Running **PyG** baseline on GNN training.
> +  Go to **`pyg_baseline/`** directory;
> + `run_GCN` in `gnn.py` set to `True` to profiling the GCN model;
> + `run_GCN` in `gnn.py` set to `False` to profiling the GIN model; 
> + `./bench` to run the script and the report 200 epoch runtime for all evaluated datasets. 

+ Running GNNAdvisor 
> +  Go to **`GNNAdvisor/`** directory
> + `run_GCN` in `gnn.py` set to `True` to profiling the GCN model;
> + `run_GCN` in `gnn.py` set to `False` to profiling the GIN model; 
> + `./bench` to run the script and the report 200 epoch runtime for all evaluated datasets. 
> +  Stand alone running with specified parameters.
>> + `--dataset`: the name of the dataset.
>> + `--dim`: the size of input embedding dimension, default: 96.
>> + `--hidden`: the size of hidden dimension, default: 16.
>> + `--classes`: the number of output classes, default: 22.
>> + `--partSize`: the size of neighbor-group, default: 32. 
>> + `--dimWorker`: the number of worker threads (MUST < 32), default: 32.
>> + `--warpPerBlock`: the number of warp per block, default: 8, recommended: GCN: 8, GIN: 2.
>> + `--loadFromTxt`: whether to load the graph TXT edge list. default: `False` (will load from npz fast).
>> + `--model`: `gcn` or `gin`. gcn has 2 layers with 16 hidden dimensions, while gin has 5 layers with 64 hidden dimensions.
>> + `--num_epoches`: the number of epoches for training, default: 200.
>> + `--manual_mode`: this a **flag** without parameter value. If this flag is specified, it will use the value from the parameter `partSize`, `dimWorker` and `dimWorker`. Otherwise, it will determine these three performance-related parameters automatically by `Decider`. Note that `Decider` will generate two different sets of parameters for input and hidden layers based on a GNN model and the dataset input characters.

**Note** that 1) accuracy evaluation are omitted; 2) the reported time per epoch only includes the GNN model forward and backward computation, excluding the data loading and some preprocessing. 
