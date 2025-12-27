"""
Main entry point for training and evaluating compact diffusion models.
"""

import os
import yaml
import argparse
import logging
from train import train
from generate import generate_images
from utils import set_seed, create_subset


def main():
    """Main function to handle different modes: train, generate, or compress."""
    parser = argparse.ArgumentParser(description="Compact Diffusion Models")
    parser.add_argument("--mode", type=str, required=True, 
                       choices=["train", "generate", "compress", "create_subset"],
                       help="Mode: train, generate, compress, or create_subset")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%I:%M:%S"
    )
    
    if args.mode == "train":
        # Training mode
        if not args.config:
            raise ValueError("--config is required for training")
        
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        logging.info(f"Starting training with config: {args.config}")
        train(config)
        
    elif args.mode == "generate":
        # Generation mode
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for generation")
        
        if args.config:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
        
        # Extract generation parameters
        num_classes = config.get("num_classes", 10)
        compress = 2 if config.get("compress", False) else 1
        image_size = config.get("image_size", 64)
        device = config.get("device", "cuda")
        
        logging.info(f"Generating images from checkpoint: {args.checkpoint}")
        
        # Generate both conditional and unconditional images
        for conditional in [True, False]:
            generate_images(
                model_checkpoint=args.checkpoint,
                num_classes=num_classes,
                compress=compress,
                device=device,
                image_size=image_size,
                batch_size=10,
                cfg_scale=0,
                results_dir="results/generated_images",
                conditional=conditional
            )
    
    elif args.mode == "compress":
        # Compression mode
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for compression")
        
        from compress import prune_model, quantize_model
        
        logging.info(f"Compressing model: {args.checkpoint}")
        
        # Prune the model
        model, pruned_checkpoint = prune_model(
            args.checkpoint,
            pruning_ratio=0.3,
            device="cuda"
        )
        
        # Quantize the pruned model
        quantized_path = pruned_checkpoint.replace(".pt", "_quantized.pt")
        quantize_model(model, quantized_path, device="cpu")
        
    elif args.mode == "create_subset":
        # Create dataset subset
        if not args.config:
            raise ValueError("--config is required for creating subset")
        
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        original_path = config.get("original_dataset_path", "data/cifar10")
        subset_path = config.get("dataset_path", "data/subset")
        subset_size = config.get("subset_size", 1000)
        
        logging.info(f"Creating subset of {subset_size} images")
        create_subset(original_path, subset_path, subset_size)


if __name__ == "__main__":
    main()
