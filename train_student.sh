#!/bin/bash

# Train student model with knowledge distillation

echo "Training Student Model with Knowledge Distillation..."
python src/main.py \
    --mode train \
    --config configs/student_config.yaml \
    --seed 42

echo "Training completed!"
echo "Model saved to: models/DDPM_student_distilled/"
