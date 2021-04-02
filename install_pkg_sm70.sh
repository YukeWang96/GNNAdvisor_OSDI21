cd GNNAdvisor/GNNConv/
TORCH_CUDA_ARCH_LIST="7.0" python setup.py clean --all install      # for Tesla V100

cd ../../

cd rabbit_module/src/
TORCH_CUDA_ARCH_LIST="7.0" python setup.py clean --all install      # for Tesla V100