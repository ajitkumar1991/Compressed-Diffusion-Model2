# Compact Diffusion Models

A PyTorch implementation of compact diffusion models with knowledge distillation, pruning, and quantization for efficient image generation.

## Overview

This project implements Denoising Diffusion Probabilistic Models (DDPM) with model compression techniques:

- **Knowledge Distillation**: Transfer learning from a large teacher model to a compact student model
- **Structured Pruning**: Remove redundant channels to reduce model size
- **Dynamic Quantization**: Reduce numerical precision for faster inference

## Features

- ✅ Conditional and unconditional image generation
- ✅ Knowledge distillation from teacher to student models
- ✅ Structured channel pruning
- ✅ Dynamic quantization
- ✅ Mixed precision training (FP16)
- ✅ Exponential Moving Average (EMA) for stable training
- ✅ Classifier-free guidance
- ✅ Checkpointing and resumable training

## Project Structure

```
compact_diffusion_project/
├── src/
│   ├── models.py          # Model architectures (UNet, Self-Attention, EMA)
│   ├── diffusion.py       # Diffusion process (forward/reverse)
│   ├── train.py           # Training logic with distillation
│   ├── generate.py        # Image generation script
│   ├── compress.py        # Pruning and quantization
│   ├── utils.py           # Utility functions
│   └── main.py            # Main entry point
├── configs/
│   ├── teacher_config.yaml  # Teacher model configuration
│   └── student_config.yaml  # Student model configuration
├── scripts/
│   ├── train_teacher.sh     # Script to train teacher
│   ├── train_student.sh     # Script to train student
│   └── compress_model.sh    # Script to compress model
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU support)
- 8GB+ GPU memory (16GB recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/compact-diffusion-models.git
cd compact-diffusion-models
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download and prepare CIFAR-10 dataset (or your own dataset):
```bash
# The dataset should be organized as:
# data/cifar10/
#   ├── class1/
#   ├── class2/
#   └── ...
```

## Quick Start

### 1. Create Dataset Subset (Optional)

```bash
python src/main.py --mode create_subset --config configs/teacher_config.yaml
```

### 2. Train Teacher Model

```bash
python src/main.py --mode train --config configs/teacher_config.yaml
```

### 3. Train Student Model with Distillation

First, update `teacher_path` in `configs/student_config.yaml` to point to your trained teacher model, then:

```bash
python src/main.py --mode train --config configs/student_config.yaml
```

### 4. Generate Images

```bash
python src/main.py --mode generate \
    --checkpoint models/DDPM_teacher/final_model.pt \
    --config configs/teacher_config.yaml
```

### 5. Compress Model (Pruning + Quantization)

```bash
python src/main.py --mode compress \
    --checkpoint models/DDPM_student_distilled/final_model.pt
```

## Detailed Usage

### Training Configuration

Edit `configs/teacher_config.yaml` or `configs/student_config.yaml`:

```yaml
run_name: "DDPM_teacher"          # Experiment name
epochs: 200                        # Number of training epochs
batch_size: 64                     # Batch size
image_size: 64                     # Image resolution
num_classes: 10                    # Number of classes
dataset_path: "data/cifar10_subset"  # Path to dataset
device: "cuda"                     # Device (cuda/cpu)
learning_rate: 3e-4               # Learning rate
save_interval: 10                 # Save checkpoint every N epochs

# Training options
mixed_precision: true             # Use FP16 training
use_ema: true                     # Use exponential moving average
distillation: false               # Enable knowledge distillation (for student)
compress: false                   # Use compressed architecture (for student)

# For student model only:
teacher_path: "models/DDPM_teacher/final_model.pt"
```

### Advanced Training

#### With Custom Data Augmentation

Modify the `get_data()` function in `src/utils.py`:

```python
transforms_list = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

#### Resume Training from Checkpoint

Training automatically resumes from the latest checkpoint if found in the checkpoint directory.

### Compression

#### Pruning Only

```bash
python src/compress.py \
    --checkpoint models/DDPM_student/final_model.pt \
    --prune \
    --pruning_ratio 0.3 \
    --device cuda
```

#### Quantization Only

```bash
python src/compress.py \
    --checkpoint models/DDPM_student/final_model.pt \
    --quantize
```

#### Both Pruning and Quantization

```bash
python src/compress.py \
    --checkpoint models/DDPM_student/final_model.pt \
    --prune \
    --pruning_ratio 0.3 \
    --quantize
```

### Image Generation

#### Generate Conditional Images

```bash
python src/generate.py \
    --checkpoint models/DDPM_teacher/final_model.pt \
    --num_classes 10 \
    --conditional \
    --device cuda
```

#### Generate with Classifier-Free Guidance

```bash
python src/generate.py \
    --checkpoint models/DDPM_teacher/final_model.pt \
    --cfg_scale 3.0 \
    --conditional
```

## Model Architecture

### U-Net Backbone

- **Encoder**: 3 downsampling blocks with self-attention
- **Bottleneck**: 3 convolutional blocks
- **Decoder**: 3 upsampling blocks with skip connections
- **Conditioning**: Timestep and class label embeddings



## Troubleshooting

### CUDA Out of Memory

- Reduce `batch_size` in config
- Enable `mixed_precision: true`
- Use gradient accumulation

### Slow Training

- Enable `mixed_precision: true`
- Increase `batch_size` if memory allows
- Use multiple GPUs (automatic with DataParallel)

### Poor Image Quality

- Train for more epochs (500+)
- Use full dataset instead of subset
- Adjust learning rate schedule
- Increase model capacity (reduce compression)



## References

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [Distilling the Knowledge in a Neural Network (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531)
- [Pruning Filters for Efficient ConvNets (Li et al., 2017)](https://arxiv.org/abs/1608.08710)


