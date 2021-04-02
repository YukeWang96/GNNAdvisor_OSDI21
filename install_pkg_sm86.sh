cd GNNAdvisor/GNNConv/
TORCH_CUDA_ARCH_LIST="8.6" python setup.py clean --all install      # for RTX3090/RTX3070

cd ../../

cd rabbit_module/src/
TORCH_CUDA_ARCH_LIST="8.6" python setup.py clean --all install      # for RTX3090/RTX3070
