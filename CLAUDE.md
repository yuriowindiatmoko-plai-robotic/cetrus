# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DATSR (Deformable Attention Transformer for Super-Resolution) is a PyTorch implementation of "Reference-based Image Super-Resolution with Deformable Attention Transformer" (ECCV 2022). The project uses reference images to enhance low-resolution input images through a novel deformable attention transformer architecture.

## Development Environment Setup

### Dependencies Installation
```bash
# Install PyTorch and CUDA toolkit
conda install pytorch=1.7.1 torchvision cudatoolkit=10.1 -c pytorch

# Install specific compatible versions
pip install timm==0.6.12
pip install mmcv-full==1.3.17 -f https://download.openmm.org/mmcv/dist/cu101/torch1.7.0/index.html
pip install numpy opencv-python scikit-image scikit-learn tqdm

# Verify mmcv.ops installation
python -c "from mmcv.ops import DeformConv2d; print('mmcv.ops working correctly!')"
```

**Important Notes**:
- Use PyTorch 1.7.1 with CUDA 10.0/10.1 for compatibility
- timm==0.6.12 is required - newer versions need torch.fx which isn't in PyTorch 1.7.1
- mmcv-full is essential for deformable convolution operations

## Common Commands

### Testing/Inference
```bash
# Test with MSE-trained model
PYTHONPATH="./:${PYTHONPATH}" python DATSR/datsr/test.py -opt "DATSR/options/test/test_restoration_mse.yml"

# Test with GAN-trained model
PYTHONPATH="./:${PYTHONPATH}" python DATSR/datsr/test.py -opt "DATSR/options/test/test_restoration_gan.yml"
```

### Training
```bash
# Train with MSE loss only
PYTHONPATH="./:${PYTHONPATH}" python DATSR/datsr/train.py -opt "DATSR/options/train/train_restoration_mse.yml"

# Train with GAN loss
PYTHONPATH="./:${PYTHONPATH}" python DATSR/datsr/train.py -opt "DATSR/options/train/train_restoration_gan.yml"
```

### Data Preparation
- Training dataset: CUFED dataset (expected in `datasets-root/train/`)
- Test datasets: CUFED5 and WR-SR (expected in `datasets/`)
- Download from links in README.md and place in appropriate directories

## Architecture Overview

### Core Components

1. **RefRestorationModel** (`DATSR/datsr/models/ref_restoration_model.py`): Main model class orchestrating the super-resolution pipeline

2. **SwinUnetv3RestorationNet** (`DATSR/datsr/models/archs/swin_unetv3_ref_restoration_arch.py`): Generator network using Swin Transformer U-Net architecture

3. **FlowSimCorrespondenceGenerationArch** (`DATSR/datsr/models/archs/flow_similarity_corres_generation_arch.py`): Handles correspondence generation between LR and reference images

4. **ContrasExtractorSep** (`DATSR/datsr/models/archs/contras_extractor_arch.py`): Extracts transformation-invariant features

### Key Architecture Patterns

- **Dynamic Module Loading**: Models and datasets are loaded dynamically using configuration strings
- **Multi-Scale Processing**: Architecture operates at multiple resolutions for better detail preservation
- **Reference-Based Processing**: Uses auxiliary reference images to enhance reconstruction quality
- **Deformable Attention**: Novel attention mechanism for better feature matching

### Data Flow

1. **Input**: Low-resolution image + reference image
2. **Feature Extraction**: TFE (Texture Feature Encoder) extracts transformation-invariant features
3. **Correspondence**: RDA (Reference-based Deformable Attention) finds relevant textures
4. **Aggregation**: RFA (Residual Feature Aggregation) combines features
5. **Output**: High-resolution super-resolved image

## Configuration System

- YAML-based configuration in `DATSR/options/`
- Separate configs for training (`train/`) and testing (`test/`)
- Model architecture, datasets, and training parameters all configurable
- Two model variants: MSE-trained (reconstruction-focused) and GAN-trained (perceptual quality)

## Working Directory Structure

```
/home/yurio/Public/PROJECT/cetrus/
├── DATSR/                    # Main implementation
│   ├── datsr/               # Core modules
│   ├── options/             # YAML configurations
│   ├── experiments/         # Model checkpoints and results
│   └── datasets/            # Dataset files and metadata
└── datasets-root/           # Raw training datasets
```

## Development Notes

- Always use `PYTHONPATH="./:${PYTHONPATH}"` when running scripts to enable imports
- Pretrained models should be placed in `DATSR/experiments/pretrained_model/`
- The codebase uses relative imports extensively, hence the PYTHONPATH requirement
- GPU training is supported via CUDA, configure with `gpu_ids` in YAML files
- TensorBoard logging is available during training