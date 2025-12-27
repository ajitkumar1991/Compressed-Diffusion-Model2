#!/bin/bash

# Train teacher model script

echo "Training Teacher Model..."
python src/main.py \
    --mode train \
    --config configs/teacher_config.yaml \
    --seed 42

echo "Training completed!"
echo "Model saved to: models/DDPM_teacher/"
