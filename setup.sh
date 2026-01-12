#!/bin/bash
# =============================================================================
# Hunyuan3D 2.1 Setup Script for NVIDIA RTX PRO 6000 Blackwell GPU
# =============================================================================
# This script automates the complete setup for Blackwell (sm_120) GPUs
# 
# Prerequisites:
# - Ubuntu 22.04+ or similar Linux distribution
# - Miniconda/Anaconda installed
# - NVIDIA driver 560+ with Blackwell support
#
# Usage: 
#   git clone https://github.com/satyam-fp/Hunyuan3D-2.1.git
#   cd Hunyuan3D-2.1
#   git checkout blackwell-gpu-support
#   chmod +x setup.sh
#   ./setup.sh
#
# =============================================================================

set -e  # Exit on any error

echo "=========================================="
echo "Hunyuan3D 2.1 Setup for Blackwell GPU"
echo "=========================================="
echo ""

# =============================================================================
# Configuration
# =============================================================================
CONDA_ENV_NAME="hunyuan3d"
PYTHON_VERSION="3.10"
CUDA_VERSION="12.8.93"
GCC_VERSION="13"
TORCH_CUDA_ARCH="12.0"  # Blackwell architecture

# Get script directory (project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Project directory: $SCRIPT_DIR"
echo ""

# =============================================================================
# Source conda
# =============================================================================
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    echo "Found Miniconda at ~/miniconda3"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    echo "Found Anaconda at ~/anaconda3"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
    echo "Found Conda at /opt/conda"
else
    echo "ERROR: Cannot find conda installation."
    echo "Please install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# =============================================================================
# Step 1: Create or activate conda environment
# =============================================================================
echo ""
echo "[Step 1/10] Setting up conda environment '$CONDA_ENV_NAME' with Python $PYTHON_VERSION..."
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "  -> Environment already exists, activating..."
else
    echo "  -> Creating new environment..."
    conda create -n "$CONDA_ENV_NAME" python=$PYTHON_VERSION -y
fi
conda activate "$CONDA_ENV_NAME"
echo "  -> Active Python: $(which python)"

# =============================================================================
# Step 2: Install PyTorch with CUDA 12.8 for Blackwell support
# =============================================================================
echo ""
echo "[Step 2/10] Installing PyTorch 2.9+ with CUDA 12.8 for Blackwell GPU support..."
echo "  -> CRITICAL: Standard PyTorch 2.5.x does NOT support Blackwell (sm_120)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --quiet

# Verify PyTorch installation
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)")
echo "  -> Installed PyTorch: $TORCH_VERSION with CUDA: $TORCH_CUDA"

# =============================================================================
# Step 3: Install pip requirements
# =============================================================================
echo ""
echo "[Step 3/10] Installing pip requirements..."
pip install -r requirements.txt --quiet
echo "  -> Requirements installed"

# =============================================================================
# Step 4: Install CUDA 12.8 toolkit via conda
# =============================================================================
echo ""
echo "[Step 4/10] Installing CUDA $CUDA_VERSION toolkit via conda..."
conda install -c nvidia cuda-nvcc=$CUDA_VERSION cuda-toolkit -y
echo "  -> nvcc version: $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')"

# =============================================================================
# Step 5: Install GCC 13 (compatible with CUDA 12.8)
# =============================================================================
echo ""
echo "[Step 5/10] Installing GCC $GCC_VERSION from conda-forge..."
echo "  -> CUDA 12.8 requires GCC <= 13"
conda install -c conda-forge gcc_linux-64=$GCC_VERSION gxx_linux-64=$GCC_VERSION -y

# =============================================================================
# Step 6: Create compiler symlinks
# =============================================================================
echo ""
echo "[Step 6/10] Creating compiler symlinks..."
ln -sf "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" "$CONDA_PREFIX/bin/gcc"
ln -sf "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" "$CONDA_PREFIX/bin/g++"
ln -sf "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" "$CONDA_PREFIX/bin/c++"
ln -sf "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc" "$CONDA_PREFIX/bin/cc"
echo "  -> GCC version: $(gcc --version | head -1)"

# =============================================================================
# Step 7: Build CUDA extensions
# =============================================================================
echo ""
echo "[Step 7/10] Building CUDA extensions for Blackwell..."
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH"

echo "  -> Building custom_rasterizer..."
cd "$SCRIPT_DIR/hy3dpaint/custom_rasterizer"
pip install --no-build-isolation -e . --quiet
cd "$SCRIPT_DIR"

echo "  -> Compiling DifferentiableRenderer..."
cd "$SCRIPT_DIR/hy3dpaint/DifferentiableRenderer"
bash compile_mesh_painter.sh
cd "$SCRIPT_DIR"
echo "  -> CUDA extensions built successfully"

# =============================================================================
# Step 8: Download RealESRGAN checkpoint
# =============================================================================
echo ""
echo "[Step 8/10] Downloading RealESRGAN checkpoint..."
mkdir -p "$SCRIPT_DIR/hy3dpaint/ckpt"
if [ ! -f "$SCRIPT_DIR/hy3dpaint/ckpt/RealESRGAN_x4plus.pth" ]; then
    wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
         -P "$SCRIPT_DIR/hy3dpaint/ckpt"
    echo "  -> Downloaded RealESRGAN_x4plus.pth"
else
    echo "  -> Checkpoint already exists"
fi

# =============================================================================
# Step 9: Fix NumPy compatibility
# =============================================================================
echo ""
echo "[Step 9/10] Downgrading NumPy for onnxruntime compatibility..."
pip install 'numpy<2' --force-reinstall --quiet
echo "  -> NumPy version: $(python -c 'import numpy; print(numpy.__version__)')"

# =============================================================================
# Step 10: Install Blender 4.5 (for GLB export)
# =============================================================================
echo ""
echo "[Step 10/10] Installing Blender 4.5..."
BLENDER_VERSION="4.5.0"
BLENDER_DIR="/opt/blender-${BLENDER_VERSION}-linux-x64"
if [ ! -d "$BLENDER_DIR" ]; then
    echo "  -> Downloading Blender $BLENDER_VERSION..."
    cd /tmp
    wget -q https://download.blender.org/release/Blender4.5/blender-${BLENDER_VERSION}-linux-x64.tar.xz
    sudo tar xf blender-${BLENDER_VERSION}-linux-x64.tar.xz -C /opt/
    rm blender-${BLENDER_VERSION}-linux-x64.tar.xz
    sudo ln -sf "$BLENDER_DIR/blender" /usr/local/bin/blender
    cd "$SCRIPT_DIR"
    echo "  -> Blender installed at $BLENDER_DIR"
else
    echo "  -> Blender already installed"
fi
echo "  -> Blender version: $(blender --version | head -1)"

# =============================================================================
# Verification
# =============================================================================
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python -c "
import torch
import sys

print(f'Python: {sys.version.split()[0]}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    cap = torch.cuda.get_device_capability(0)
    print(f'Compute capability: {cap[0]}.{cap[1]}')

try:
    import custom_rasterizer_kernel
    print('✓ custom_rasterizer OK')
except ImportError as e:
    print(f'✗ custom_rasterizer failed: {e}')
"

# =============================================================================
# Done
# =============================================================================
echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "Usage:"
echo "  conda activate $CONDA_ENV_NAME"
echo "  python run_texture_gen.py --mesh <mesh> --image <image> --prompt 'your prompt' --output ./outputs"
echo ""
