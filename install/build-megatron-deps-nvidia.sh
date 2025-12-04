#!/bin/bash

set -e
    
CUDA_TAG="cu$(nvcc --version | grep "Cuda compilation tools" | awk '{print $5}' | cut -d '.' -f 1,2 | tr -d '.,')"
uv pip install torch torchaudio torchvision --extra-index-url https://download.pytorch.org/whl/${CUDA_TAG}

# flash-attention
cu=$(nvcc --version | grep "Cuda compilation tools" | awk '{print $5}' | cut -d '.' -f 1)
torch=$(pip show torch | grep Version | awk '{print $2}' | cut -d '+' -f 1 | cut -d '.' -f 1,2)
cp=$(python3 --version | awk '{print $2}' | awk -F. '{print $1$2}')
cxx=$(g++ --version | grep 'g++' | awk '{print $3}' | cut -d '.' -f 1)
flash_attn_version="2.8.0.post2"
wget --continue --timeout=60 --no-check-certificate --tries=5 --waitretry=10 https://github.com/Dao-AILab/flash-attention/releases/download/v${flash_attn_version}/flash_attn-${flash_attn_version}+cu${cu}torch${torch}cxx${cxx}abiFALSE-cp${cp}-cp${cp}-linux_x86_64.whl
uv pip install --no-cache-dir flash_attn-${flash_attn_version}+cu${cu}torch${torch}cxx${cxx}abiFALSE-cp${cp}-cp${cp}-linux_x86_64.whl --index-url https://mirrors.aliyun.com/pypi/simple/
rm flash_attn-${flash_attn_version}+cu${cu}torch${torch}cxx${cxx}abiFALSE-cp${cp}-cp${cp}-linux_x86_64.whl

# transformer engine install for megatron-lm
git clone --recursive https://github.com/NVIDIA/TransformerEngine.git
cd TransformerEngine
git checkout e9a5fa4e  # Date:   Thu Sep 4 22:39:53 2025 +0200
uv pip install --no-build-isolation . --index-url https://mirrors.aliyun.com/pypi/simple/
cd ..
rm -r ./TransformerEngine    

# apex install for megatron-lm
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings '--build-option=--cpp_ext' --config-settings '--build-option=--cuda_ext' ./ --index-url https://mirrors.aliyun.com/pypi/simple/
cd ..
rm -r ./apex

