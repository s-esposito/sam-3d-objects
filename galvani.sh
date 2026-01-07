# conda create -n sam3d-objects python=3.11 -y
# conda install nvidia::cuda-toolkit==12.8.1
# pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
# # conda install -c iopath iopath
# pip install scikit-image matplotlib imageio plotly opencv-python
# pip install git+https://github.com/nerfstudio-project/gsplat.git

# conda create -n sam3d-objects python=3.11 -y
conda env create -f environments/default.yml
# export CUDA_VER=12.1 ; export CUDA_HOME=/usr/local/cuda-$CUDA_VER ; export CUDA_PATH=/usr/local/cuda-$CUDA_VER ; export PATH=/usr/local/cuda-$CUDA_VER/bin:$PATH ; export LD_LIBRARY_PATH=/usr/local/cuda-$CUDA_VER/lib64:/usr/local/cuda-$CUDA_VER/extras/CUPTI/lib64:$LD_LIBRARY_PATH
# pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
pip install -e '.[dev]'

# conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# https://anaconda.org/channels/pytorch3d/packages/pytorch3d/files?name=py311_cu121_pyt241

# pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
# conda install pytorch3d-0.7.8-py311_cu121_pyt241.tar.bz2

# echo "Installing pytorch3d..."
# Solution: Use system CUDA toolkit instead of conda's to avoid fatbinary crashes
# CUDA_HOME points to system CUDA, TORCH_CUDA_ARCH_LIST specifies GPU arch, MAX_JOBS=1 prevents resource exhaustion
CUDA_HOME=/usr/local/cuda-12.1 TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation

# pip install -e '.[p3d]' # pytorch3d dependency on pytorch is broken, this 2-step approach solves it

# Galvani has 2.28
# flash-attn installation (OPTIONAL - can skip if GLIBC < 2.32)
# Note: flash-attn 2.8.3 pre-built wheel requires GLIBC 2.32+, but CentOS 7 has GLIBC 2.28
# Building from source still links against conda's GLIBC 2.32+
# Workaround: Skip installation and use environment variables to force sdpa backend:
#   export ATTN_BACKEND=sdpa
#   export SPARSE_ATTN_BACKEND=sdpa

# conda install -c conda-forge gcc_linux-64 gxx_linux-64
# conda install -c conda-forge glibc=2.32
# Or try installing (will fail on systems with GLIBC < 2.32):
# CUDA_HOME=/usr/local/cuda-12.1 TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=2 pip install flash-attn==2.8.3 --no-build-isolation

# for inference
# export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.1_cu121.html"
pip install -e '.[inference]'

# echo "Installing gsplat..."
CUDA_HOME=/usr/local/cuda-12.1 TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=1 pip install git+https://github.com/nerfstudio-project/gsplat.git --no-build-isolation

# patch things that aren't yet in official pip packages
./patching/hydra # https://github.com/facebookresearch/hydra/pull/2863

pip install 'huggingface-hub[cli]<1.0'

pip install open3d
pip install xformers