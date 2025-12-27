"""
Model compression utilities: pruning and quantization.
"""

import os
import torch
import torch.nn as nn
import logging
from models import UNet_conditional, SelfAttention
from utils import inspect_model, print_size_of_model


def prune_model(model_checkpoint, pruning_ratio=0.3, device="cuda", num_classes=10, compress=2):
    """
    Apply structured pruning to a model using torch_pruning.
    
    Args:
        model_checkpoint (str): Path to model checkpoint
        pruning_ratio (float): Fraction of channels to prune (0-1)
        device (str): Device to use
        num_classes (int): Number of classes
        compress (int): Compression factor
        
    Returns:
        Pruned model
    """
    try:
        import torch_pruning as tp
    except ImportError:
        raise ImportError("Please install torch-pruning: pip install torch-pruning")
    
    logging.info(f"Pruning model from checkpoint: {model_checkpoint}")
    
    # Load model
    model = inspect_model(
        model_checkpoint, 
        UNet_conditional, 
        num_classes=num_classes, 
        compress=compress, 
        device=device
    )
    model.eval()

    # Define example inputs
    example_inputs = (
        torch.randn(1, 3, 64, 64).to(device),
        torch.ones(1).long().to(device),
        torch.tensor([0]).long().to(device)
    )

    # Initialize pruning importance metric
    imp = tp.importance.MagnitudeImportance()

    # Identify layers to ignore (e.g., self-attention layers)
    ignored_layers = []
    for module in model.modules():
        if isinstance(module, SelfAttention):
            ignored_layers.append(module)

    # Initialize pruner
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=1,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
        round_to=4  # Ensure channel counts are multiples of 4
    )

    # Apply pruning
    logging.info(f"Applying {pruning_ratio*100}% pruning...")
    pruner.step(interactive=False)

    logging.info("Pruned model size:")
    print_size_of_model(model)

    # Save pruned model
    pruned_checkpoint = model_checkpoint.replace(".pt", f"_pruned_{pruning_ratio}.pt")
    torch.save(model.state_dict(), pruned_checkpoint)
    logging.info(f"Pruned model saved to: {pruned_checkpoint}")

    return model, pruned_checkpoint


def quantize_model(model, quantized_checkpoint_path, device="cpu"):
    """
    Apply dynamic quantization to a model.
    
    Args:
        model: PyTorch model to quantize
        quantized_checkpoint_path (str): Path to save quantized model
        device (str): Device (should be 'cpu' for quantization)
        
    Returns:
        Quantized model
    """
    logging.info("Quantizing model...")
    
    # Move model to CPU
    model.to(device)
    model.eval()

    logging.info("Before quantization:")
    print_size_of_model(model)

    # Apply dynamic quantization
    model_quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )

    logging.info("After quantization:")
    print_size_of_model(model_quantized)

    # Save quantized model
    torch.save(model_quantized.state_dict(), quantized_checkpoint_path)
    logging.info(f"Quantized model saved to: {quantized_checkpoint_path}")

    return model_quantized


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compress diffusion model via pruning and quantization")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prune", action="store_true", help="Apply pruning")
    parser.add_argument("--pruning_ratio", type=float, default=0.3, help="Pruning ratio")
    parser.add_argument("--quantize", action="store_true", help="Apply quantization")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--compress", type=int, default=2, help="Compression factor")
    parser.add_argument("--device", type=str, default="cuda", help="Device for pruning")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    current_checkpoint = args.checkpoint
    
    # Apply pruning if requested
    if args.prune:
        model, current_checkpoint = prune_model(
            current_checkpoint,
            pruning_ratio=args.pruning_ratio,
            device=args.device,
            num_classes=args.num_classes,
            compress=args.compress
        )
    
    # Apply quantization if requested
    if args.quantize:
        if not args.prune:
            # Load model if we didn't just prune it
            model = inspect_model(
                current_checkpoint,
                UNet_conditional,
                num_classes=args.num_classes,
                compress=args.compress,
                device="cpu"
            )
        
        quantized_path = current_checkpoint.replace(".pt", "_quantized.pt")
        quantize_model(model, quantized_path, device="cpu")
