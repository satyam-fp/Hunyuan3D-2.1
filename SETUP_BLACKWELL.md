# Hunyuan3D 2.1 Setup Guide for NVIDIA Blackwell GPUs

Complete setup guide for running Hunyuan3D 2.1 on NVIDIA RTX PRO 6000 Blackwell GPUs (sm_120/compute_12.0 architecture).

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Setup (Automated)](#quick-setup-automated)
- [Manual Setup Steps](#manual-setup-steps)
- [Verification](#verification)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Environment Summary](#environment-summary)

## Prerequisites

| Requirement | Version/Details |
|------------|-----------------|
| OS | Ubuntu 22.04+ or similar Linux |
| GPU | NVIDIA RTX PRO 6000 Blackwell (or other Blackwell GPU) |
| Driver | NVIDIA driver 560+ with Blackwell support |
| Conda | Miniconda3 or Anaconda3 |

## Quick Setup (Automated)

```bash
# Clone the repository
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
cd Hunyuan3D-2.1

# Run the automated setup script
chmod +x setup.sh
./setup.sh
```

The script takes approximately 10-15 minutes and handles all steps automatically.

## Manual Setup Steps

### Step 1: Create Conda Environment

```bash
conda create -n hunyuan3d python=3.10 -y
conda activate hunyuan3d
```

### Step 2: Install PyTorch with CUDA 12.8

> **⚠️ CRITICAL**: Standard PyTorch 2.5.x does NOT support Blackwell GPUs (sm_120). You MUST use PyTorch 2.7+ with CUDA 12.8.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify installation:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
# Expected: PyTorch: 2.9.1+cu128, CUDA: 12.8
```

### Step 3: Patch requirements.txt

The `bpy` (Blender Python) module is not pip-installable and must be commented out:

```bash
sed -i 's/^bpy==.*/# bpy==4.0  # Commented out - requires Blender/' requirements.txt
```

### Step 4: Install Python Requirements

```bash
pip install -r requirements.txt
```

### Step 5: Install CUDA 12.8 Toolkit

The nvcc compiler must match PyTorch's CUDA version:

```bash
conda install -c nvidia cuda-nvcc=12.8.93 cuda-toolkit -y
```

Verify:
```bash
nvcc --version
# Should show: Cuda compilation tools, release 12.8, V12.8.93
```

### Step 6: Install GCC 13

> **Note**: CUDA 12.8 only supports GCC ≤ 13. GCC 14+ will cause compilation errors.

```bash
conda install -c conda-forge gcc_linux-64=13 gxx_linux-64=13 -y
```

### Step 7: Create Compiler Symlinks

The conda GCC uses prefixed names that need symlinks:

```bash
ln -sf $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc $CONDA_PREFIX/bin/gcc
ln -sf $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ $CONDA_PREFIX/bin/g++
ln -sf $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++ $CONDA_PREFIX/bin/c++
ln -sf $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc $CONDA_PREFIX/bin/cc
```

### Step 8: Build CUDA Extensions

```bash
# Set environment variables
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="12.0"  # Blackwell architecture

# Build custom_rasterizer
cd hy3dpaint/custom_rasterizer
pip install --no-build-isolation -e .
cd ../..

# Compile mesh_painter
cd hy3dpaint/DifferentiableRenderer
bash compile_mesh_painter.sh
cd ../..
```

### Step 9: Download RealESRGAN Checkpoint

```bash
mkdir -p hy3dpaint/ckpt
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
     -P hy3dpaint/ckpt
```

### Step 10: Apply Code Patches

#### Patch 1: Fix RealESRGAN Path

In `hy3dpaint/textureGenPipeline.py`, change line 45:
```python
# From:
self.realesrgan_ckpt_path = "ckpt/RealESRGAN_x4plus.pth"
# To:
self.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
```

Or use sed:
```bash
sed -i 's|self\.realesrgan_ckpt_path = "ckpt/RealESRGAN_x4plus.pth"|self.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"|' \
    hy3dpaint/textureGenPipeline.py
```

#### Patch 2: Make bpy Import Lazy

Replace `hy3dpaint/DifferentiableRenderer/mesh_utils.py` with the patched version from `patches/mesh_utils_patched.py`, or manually:

1. Remove `import bpy` from line 17
2. Add comment: `# bpy is imported lazily in convert_obj_to_glb since it requires Blender`
3. Modify `convert_obj_to_glb()` function to import bpy inside the function with try/except

### Step 11: Downgrade NumPy

onnxruntime is compiled against NumPy 1.x and crashes with NumPy 2.x:

```bash
pip install 'numpy<2' --force-reinstall
```

## Verification

```bash
conda activate hunyuan3d

# Check PyTorch and CUDA
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Compute Capability: {torch.cuda.get_device_capability(0)}')
"

# Test custom_rasterizer
python -c "import custom_rasterizer_kernel; print('custom_rasterizer OK')"

# Run texture generation test
python run_texture_gen.py --mesh assets/1.glb --image assets/demo.png \
    --prompt "high quality" --output ./outputs --no_glb
```

Expected verification output:
```
PyTorch: 2.9.1+cu128
CUDA: 12.8
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
Compute Capability: (12, 0)
```

## Usage

### Texture Generation Script

```bash
python run_texture_gen.py \
    --mesh assets/1.glb \
    --image assets/demo.png \
    --prompt "high quality 3d model with detailed textures" \
    --output ./outputs \
    --no_glb
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--mesh` | (required) | Input mesh path (OBJ/GLB) |
| `--image` | (required) | Conditioning image for texture style |
| `--prompt` | "high quality" | Text prompt for texture generation |
| `--output` | ./outputs | Output directory |
| `--max_views` | 6 | Number of multiview angles |
| `--resolution` | 512 | Generation resolution |
| `--no_glb` | False | Skip GLB export (use if Blender not installed) |

### Output Files

| File | Description |
|------|-------------|
| `*_textured.obj` | Textured mesh in OBJ format |
| `*_textured.mtl` | Material file |
| `*_textured.jpg` | Albedo/diffuse texture |
| `*_textured_metallic.jpg` | Metallic PBR map |
| `*_textured_roughness.jpg` | Roughness PBR map |

### Gradio Web Interface

```bash
python gradio_app.py \
    --model_path tencent/Hunyuan3D-2.1 \
    --subfolder hunyuan3d-dit-v2-1 \
    --texgen_model_path tencent/Hunyuan3D-2.1 \
    --low_vram_mode
```

## Troubleshooting

### Error: "CUDA error: no kernel image is available"

**Cause**: PyTorch version doesn't support Blackwell (sm_120).

**Solution**: Install PyTorch 2.7+ with CUDA 12.8:
```bash
pip install --upgrade --force-reinstall torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
```

### Error: "unsupported GNU version! gcc versions later than 13 are not supported"

**Cause**: GCC version too new for CUDA 12.8.

**Solution**: Install GCC 13:
```bash
conda install -c conda-forge gcc_linux-64=13 gxx_linux-64=13 -y
# Recreate symlinks
ln -sf $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc $CONDA_PREFIX/bin/gcc
```

### Error: "CUDA version mismatch"

**Cause**: nvcc version doesn't match PyTorch CUDA version.

**Solution**: Install matching CUDA toolkit:
```bash
conda install -c nvidia cuda-nvcc=12.8.93 cuda-toolkit -y
```

### Error: "numpy.core.multiarray failed to import" or "_ARRAY_API not found"

**Cause**: NumPy 2.x incompatible with onnxruntime.

**Solution**: Downgrade NumPy:
```bash
pip install 'numpy<2' --force-reinstall
```

### Error: "No module named 'bpy'"

**Cause**: Blender Python not installed.

**Solution**: Use `--no_glb` flag, or install Blender system-wide.

### Error: "gcc: No such file or directory"

**Cause**: Compiler symlinks not created.

**Solution**: Create symlinks:
```bash
ln -sf $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc $CONDA_PREFIX/bin/gcc
ln -sf $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ $CONDA_PREFIX/bin/g++
ln -sf $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++ $CONDA_PREFIX/bin/c++
```

## Environment Summary

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.10 | Required by Hunyuan3D |
| PyTorch | 2.9.1+cu128 | Minimum 2.7 for Blackwell |
| CUDA Toolkit | 12.8.93 | Must match PyTorch CUDA |
| GCC | 13.4.0 | Maximum supported by CUDA 12.8 |
| NumPy | 1.26.4 | Must be < 2.0 |
| TORCH_CUDA_ARCH_LIST | 12.0 | Blackwell compute capability |

## File Structure

```
Hunyuan3D-2.1/
├── setup.sh                    # Automated setup script
├── SETUP_BLACKWELL.md          # This documentation
├── run_texture_gen.py          # Texture generation CLI
├── patches/
│   └── mesh_utils_patched.py   # Patched mesh_utils with lazy bpy
├── hy3dpaint/
│   ├── ckpt/
│   │   └── RealESRGAN_x4plus.pth
│   ├── custom_rasterizer/      # CUDA extension (built)
│   ├── DifferentiableRenderer/
│   │   ├── mesh_utils.py       # Patched for lazy bpy import
│   │   └── mesh_painter.so     # Compiled C++ library
│   └── textureGenPipeline.py   # Patched RealESRGAN path
└── outputs/                    # Generated textures
```
