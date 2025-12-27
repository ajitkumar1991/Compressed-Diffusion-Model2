#!/bin/bash

# Compress student model via pruning and quantization

CHECKPOINT="models/DDPM_student_distilled/final_model.pt"

echo "Compressing model: $CHECKPOINT"
python src/compress.py \
    --checkpoint $CHECKPOINT \
    --prune \
    --pruning_ratio 0.3 \
    --quantize \
    --device cuda

echo "Compression completed!"
echo "Pruned model saved with '_pruned_0.3' suffix"
echo "Quantized model saved with '_quantized' suffix"
