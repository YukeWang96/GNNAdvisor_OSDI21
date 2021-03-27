cd GNNAdvisor/GNNConv/
# TORCH_CUDA_ARCH_LIST="8.6" python setup.py install      # for RTX3090
TORCH_CUDA_ARCH_LIST="6.1" python setup.py install      # for Quadro P6000

cd ../../

cd rabbit_module/src/
# TORCH_CUDA_ARCH_LIST="8.6" python setup.py install      # for RTX3090
TORCH_CUDA_ARCH_LIST="6.1" python setup.py install      # for Quadro P6000
