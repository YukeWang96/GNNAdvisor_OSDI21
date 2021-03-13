# GNNAdvisor 

## dependency 
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

+ Install [`pytorch-geometric (PyG)`](https://github.com/rusty1s/pytorch_geometric).
```
CUDA=cu111
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-geometric
```

## Node Reordering.
* [Done] adding the IPDPS reordering tool into our design.
* [Done] make a pytorch operator for this algorithm.
* [Done] merge into our Pytorch Flow.

## Parameters.  -->  Parmeter object.
* averaged edge degree.
* average edge span.
* node dimension.

## GNNAdvisor execution flow by taking all these parameter elements.
* DGL baseline [GCN and GIN]
* PyG baseline [GCN and GIN]
* Our GCN (2-layer 16 hidden)
* Our GIN (5-layer 64 hidden)