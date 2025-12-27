# Example: Using the Compact Diffusion Models Package

This notebook demonstrates how to use the reorganized code structure.

## Setup

```python
import sys
sys.path.append('src')

import torch
from models import UNet_conditional
from diffusion import Diffusion
from utils import set_seed, plot_images
```

## 1. Load a Trained Model

```python
# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model
model = UNet_conditional(
    compress=1,  # 1 for teacher, 2 for student
    num_classes=10
).to(device)

# Load checkpoint
checkpoint_path = "models/DDPM_teacher/final_model.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)

# Handle state dict
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()
print("Model loaded successfully!")
```

## 2. Generate Images

```python
# Initialize diffusion
diffusion = Diffusion(img_size=64, device=device)

# Generate conditional images (one per class)
labels = torch.arange(10).long().to(device)

generated_images = diffusion.sample(
    model=model,
    total_images=10,
    batch_size=5,
    labels=labels,
    cfg_scale=0
)

# Convert to float for visualization
images_float = [img.float() / 255.0 for img in generated_images]

# Display
plot_images(images_float, title="Generated CIFAR-10 Images")
```

## 3. Train a New Model (Interactive)

```python
import yaml
from train import train

# Load config
with open('configs/teacher_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Modify for quick test
config['epochs'] = 2
config['batch_size'] = 16
config['dataset_path'] = 'data/cifar10_subset'

# Train
train(config)
```

## 4. Compress a Model

```python
from compress import prune_model, quantize_model

# Prune
pruned_model, pruned_checkpoint = prune_model(
    model_checkpoint="models/DDPM_student/final_model.pt",
    pruning_ratio=0.3,
    device=device
)

# Quantize
quantized_checkpoint = pruned_checkpoint.replace(".pt", "_quantized.pt")
quantized_model = quantize_model(
    pruned_model,
    quantized_checkpoint,
    device="cpu"
)

print(f"Compressed model saved to: {quantized_checkpoint}")
```

## 5. Inspect Model Architecture

```python
from utils import inspect_model, print_size_of_model

# Inspect teacher
print("=== Teacher Model ===")
teacher = inspect_model(
    "models/DDPM_teacher/final_model.pt",
    UNet_conditional,
    num_classes=10,
    compress=1,
    device=device
)

# Inspect student
print("\n=== Student Model ===")
student = inspect_model(
    "models/DDPM_student/final_model.pt",
    UNet_conditional,
    num_classes=10,
    compress=2,
    device=device
)

# Compare sizes
print("\n=== Size Comparison ===")
print("Teacher:", end=" ")
print_size_of_model(teacher)
print("Student:", end=" ")
print_size_of_model(student)
reduction = (1 - (sum(p.numel() for p in student.parameters()) / 
                   sum(p.numel() for p in teacher.parameters()))) * 100
print(f"Size reduction: {reduction:.1f}%")
```

## 6. Generate with Different Settings

```python
# Generate unconditional images
print("Generating unconditional images...")
uncond_images = diffusion.sample(
    model=model,
    total_images=8,
    batch_size=8,
    labels=None,  # No labels = unconditional
    cfg_scale=0
)

# Generate with classifier-free guidance
print("Generating with CFG scale=3...")
cfg_images = diffusion.sample(
    model=model,
    total_images=10,
    batch_size=5,
    labels=labels,
    cfg_scale=3.0  # Higher = more faithful to conditioning
)

# Display
images_float = [img.float() / 255.0 for img in cfg_images]
plot_images(images_float, title="Generated with CFG")
```

## 7. Monitor Training Progress

```python
import matplotlib.pyplot as plt

# Load loss history from checkpoint
checkpoint = torch.load("checkpoints/DDPM_teacher/ckpt_epoch_100.pt")
loss_history = checkpoint.get('loss_history', [])

# Plot
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss History')
plt.grid(True)
plt.show()

print(f"Final loss: {loss_history[-1]:.6f}")
print(f"Best loss: {min(loss_history):.6f} at epoch {loss_history.index(min(loss_history)) + 1}")
```

## -----


For more examples, see the scripts in `scripts/` or read `GETTING_STARTED.md`.
