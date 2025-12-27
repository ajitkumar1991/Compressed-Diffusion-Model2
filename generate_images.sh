#!/bin/bash

# Generate images from trained model

CHECKPOINT=${1:-"models/DDPM_teacher/final_model.pt"}

echo "Generating images from: $CHECKPOINT"
python src/generate.py \
    --checkpoint $CHECKPOINT \
    --num_classes 10 \
    --image_size 64 \
    --batch_size 10 \
    --device cuda \
    --conditional

echo "Image generation completed!"
echo "Images saved to: results/generated_images/"
