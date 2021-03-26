cd GNNAdvisor/GNNConv/
TORCH_CUDA_ARCH_LIST="8.6" python setup.py install

cd ../../

cd rabbit_module/src/
TORCH_CUDA_ARCH_LIST="8.6" python setup.py install