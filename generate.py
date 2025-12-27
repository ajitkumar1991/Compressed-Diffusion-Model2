"""
Image generation script for trained diffusion models.
"""

import os
import torch
import logging
from diffusion import Diffusion
from models import UNet_conditional
from utils import inspect_model, plot_images, save_images


def generate_images(
    model_checkpoint,
    num_classes=10,
    compress=1,
    device="cuda",
    image_size=64,
    batch_size=10,
    cfg_scale=0,
    results_dir="results/generated_images",
    conditional=True
):
    """
    Generate and display images using a trained model.
    
    Args:
        model_checkpoint (str): Path to model checkpoint
        num_classes (int): Number of classes
        compress (int): Model compression factor
        device (str): Device to use
        image_size (int): Size of generated images
        batch_size (int): Batch size for generation
        cfg_scale (float): Classifier-free guidance scale
        results_dir (str): Directory to save results
        conditional (bool): Whether to generate conditional images
    """
    logging.basicConfig(level=logging.INFO)
    
    # Load model
    model = inspect_model(
        model_checkpoint, 
        UNet_conditional, 
        num_classes=num_classes, 
        compress=compress, 
        device=device
    )

    # Initialize diffusion
    diffusion = Diffusion(img_size=image_size, device=device)

    # Prepare labels
    if conditional:
        labels = torch.arange(num_classes).long().to(device)
        title_str = "Conditional Generated Images"
        subdir = "conditional"
    else:
        labels = None
        title_str = "Unconditional Generated Images"
        subdir = "unconditional"

    # Generate images (returns uint8 tensors)
    logging.info(f"Generating {num_classes} images...")
    generated_images_uint8 = diffusion.sample(
        model, 
        total_images=num_classes, 
        batch_size=batch_size, 
        labels=labels, 
        cfg_scale=cfg_scale
    )

    # Convert to float for plotting
    generated_images_float = [img.float() / 255.0 for img in generated_images_uint8]

    # Plot images
    plot_images(generated_images_float, title=title_str)

    # Save images
    save_path = os.path.join(results_dir, subdir)
    save_images(generated_images_uint8, save_path)
    
    logging.info(f"Generation complete! Images saved to {save_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate images from trained diffusion model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--compress", type=int, default=1, help="Compression factor")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--image_size", type=int, default=64, help="Image size")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("--cfg_scale", type=float, default=0, help="Classifier-free guidance scale")
    parser.add_argument("--results_dir", type=str, default="results/generated_images", help="Results directory")
    parser.add_argument("--conditional", action="store_true", help="Generate conditional images")
    
    args = parser.parse_args()
    
    generate_images(
        model_checkpoint=args.checkpoint,
        num_classes=args.num_classes,
        compress=args.compress,
        device=args.device,
        image_size=args.image_size,
        batch_size=args.batch_size,
        cfg_scale=args.cfg_scale,
        results_dir=args.results_dir,
        conditional=args.conditional
    )
