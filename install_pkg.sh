cd GNNAdvisor/GNNConv/
TORCH_CUDA_ARCH_LIST="8.6" python setup.py clean --all install      # for RTX3090/RTX3070
# TORCH_CUDA_ARCH_LIST="7.0" python setup.py clean --all install      # for Tesla V100
# TORCH_CUDA_ARCH_LIST="6.1" python setup.py clean --all install      # for Quadro P6000

cd ../../

cd rabbit_module/src/
TORCH_CUDA_ARCH_LIST="8.6" python setup.py clean --all install      # for RTX3090/RTX3070
# TORCH_CUDA_ARCH_LIST="7.0" python setup.py clean --all install      # for Tesla V100
# TORCH_CUDA_ARCH_LIST="6.1" python setup.py clean --all install      # for Quadro P6000