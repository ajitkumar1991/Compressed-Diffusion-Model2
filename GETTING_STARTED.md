# Getting Started with Compact Diffusion Models

##  Organization

Code is reorganized into following modular structure:

```
compact_diffusion_project/
├── src/                    # Source code
│   ├── models.py          # Neural network architectures
│   ├── diffusion.py       # Diffusion process implementation
│   ├── train.py           # Training loop
│   ├── generate.py        # Image generation
│   ├── compress.py        # Model compression (pruning/quantization)
│   ├── utils.py           # Helper functions
│   └── main.py            # Main entry point
├── configs/               # Configuration files
│   ├── teacher_config.yaml
│   └── student_config.yaml
├── scripts/               # Convenience scripts
│   ├── train_teacher.sh
│   ├── train_student.sh
│   ├── compress_model.sh
│   └── generate_images.sh
├── requirements.txt       # Dependencies
├── README.md             # Full documentation
├── LICENSE               # MIT License
└── .gitignore           # Git ignore rules
```

##  Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
cd compact_diffusion_project
pip install -r requirements.txt
```

### Step 2: Prepare  Data

Organize your dataset in this structure:
```
data/cifar10/
├── class1/
│   ├── img1.png
│   ├── img2.png
│   └── ...
├── class2/
│   └── ...
└── ...
```

Or create a subset:
```bash
python src/main.py --mode create_subset --config configs/teacher_config.yaml
```

### Step 3: Train the Model

```bash
# Option A: Using script
bash scripts/train_teacher.sh

# Option B: Direct command
python src/main.py --mode train --config configs/teacher_config.yaml
```

## Complete Workflow

### 1. Train Teacher Model (Large, High Quality)

```bash
# Edit configs/teacher_config.yaml to set the data path
# Then train:
python src/main.py --mode train --config configs/teacher_config.yaml
```

**Expected output:**
- Checkpoints saved to: `checkpoints/DDPM_teacher/`
- Final model: `models/DDPM_teacher/final_model.pt`
- Training takes: ~2-3 hours on V100 GPU (200 epochs)

### 2. Train Student Model (Small, via Distillation)

```bash
# Update teacher_path in configs/student_config.yaml
# Point it to: models/DDPM_teacher/final_model.pt
python src/main.py --mode train --config configs/student_config.yaml
```

**Benefits:**
- 74% smaller than teacher
- Faster inference
- Similar quality (with proper training)

### 3. Generate Images

```bash
# Generate from teacher model
python src/generate.py \
    --checkpoint models/DDPM_teacher/final_model.pt \
    --num_classes 10 \
    --conditional \
    --device cuda

# Or use the convenience script
bash scripts/generate_images.sh models/DDPM_teacher/final_model.pt
```

**Output:**
- Images will be saved to: `results/generated_images/`
- Both conditional and unconditional samples

### 4. Compress Model (Optional)

```bash
# Prune + Quantize the student model
python src/compress.py \
    --checkpoint models/DDPM_student_distilled/final_model.pt \
    --prune \
    --pruning_ratio 0.3 \
    --quantize

# Or use the script
bash scripts/compress_model.sh
```



## 🔧 Key Configuration Options

### `configs/teacher_config.yaml`

```yaml
run_name: "DDPM_teacher"       # Experiment name
epochs: 200                     # Training duration
batch_size: 64                  # Batch size (reduce if OOM)
image_size: 64                  # Image resolution
dataset_path: "data/cifar10"    # Your data location
mixed_precision: true           # Enable FP16 (faster, less memory)
use_ema: true                   # Stabilize training
```

### `configs/student_config.yaml`

```yaml
distillation: true              # Enable knowledge distillation
teacher_path: "path/to/teacher.pt"  # Trained teacher model
compress: true                  # Use 2× compression
```

## Some Common Issues & Solutions

### Issue 1: CUDA Out of Memory

**Solution:**
```yaml
# In the config file:
batch_size: 32  # or even 16
mixed_precision: true
```

### Issue 2: ImportError for torch_pruning

**Solution:**
```bash
pip install torch-pruning
```

### Issue 3: Poor Image Quality

**Causes:**
- Not enough training epochs (need 500+)
- Too small dataset (need 50k+ images)
- Too aggressive compression

**Solutions:**
```yaml
# Increase epochs
epochs: 500

# Use full dataset (not subset)
dataset_path: "data/full_cifar10"

# Reduce compression
compress: false  # or use compress: true with less pruning
```

### Issue 4: Training Taking Too Long

**Solutions:**
- Enable mixed precision: `mixed_precision: true`
- Use multiple GPUs (automatic with DataParallel)
- Reduce image size: `image_size: 32` (if acceptable)
- Reduce diffusion steps (edit `diffusion.py`)

## --Monitoring Training

### Watch Training Progress

```bash
# Terminal shows:
# - Current epoch
# - Batch-wise MSE loss
# - ETA
```

### Check Saved Checkpoints

```bash
ls checkpoints/DDPM_teacher/
# Output:
# ckpt_epoch_10.pt
# ckpt_epoch_20.pt
# ...
```

### Resume Interrupted Training

Training automatically resumes from the latest checkpoint! Just run:
```bash
python src/main.py --mode train --config configs/teacher_config.yaml
```

## 🎨 Advanced Usage

### Custom Data Augmentation

Edit `src/utils.py`, function `get_data()`:

```python
transforms_list = transforms.Compose([
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.RandomHorizontalFlip(),      # Add this
    transforms.RandomRotation(15),          # Add this
    transforms.ColorJitter(0.2, 0.2, 0.2),  # Add this
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

### Classifier-Free Guidance (Better Quality)

```python
# In generate.py, increase cfg_scale
generate_images(
    model_checkpoint="path/to/model.pt",
    cfg_scale=3.0,  # Increase from 0 to 3-7
    conditional=True
)
```

### Multi-GPU Training

Automatically enabled! DataParallel distributes across all available GPUs.

## 📁 Output Files

### After Training
```
checkpoints/DDPM_teacher/
├── ckpt_epoch_10.pt       # Periodic checkpoints
├── ckpt_epoch_20.pt
└── ema_ckpt_epoch_10.pt   # EMA checkpoints

models/DDPM_teacher/
└── final_model.pt         # Final trained model

results/DDPM_teacher/
└── loss_history.png       # Training loss plot
```

### After Generation
```
results/generated_images/
├── conditional/
│   ├── 0.png
│   ├── 1.png
│   └── ...
└── unconditional/
    ├── 0.png
    └── ...
```


## quick Steps

1. **Test the code**: Run `train_teacher.sh` with a small subset
2. **Customize configs**: Adjust hyperparameters for your needs
3. **Add features**: The modular structure makes it easy
4. **Share on GitHub**: Everything is ready for version control



