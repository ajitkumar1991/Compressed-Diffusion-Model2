"""
Utility functions for data handling, visualization, and model inspection.
"""

import os
import shutil
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_subset(original_data_path, subset_data_path, subset_size=1000):
    """
    Create a subset of data by copying a specified number of images.
    
    Args:
        original_data_path (str): Path to original dataset
        subset_data_path (str): Path to save subset
        subset_size (int): Total number of images in subset
    """
    if not os.path.exists(original_data_path):
        raise FileNotFoundError(f"Original data path '{original_data_path}' does not exist.")

    # Remove existing subset if present
    if os.path.exists(subset_data_path):
        shutil.rmtree(subset_data_path)
    os.makedirs(subset_data_path)

    # Get dataset classes
    dataset = ImageFolder(root=original_data_path)
    classes = dataset.classes
    class_to_idx = dataset.class_to_idx

    # Create class-wise subfolders
    for class_name in classes:
        os.makedirs(os.path.join(subset_data_path, class_name), exist_ok=True)

    # Sample images per class
    num_classes = len(classes)
    subset_size_per_class = subset_size // num_classes

    for class_name, idx in class_to_idx.items():
        class_folder = os.path.join(original_data_path, class_name)
        images = [
            os.path.join(class_folder, img) 
            for img in os.listdir(class_folder) 
            if img.endswith(('.png', '.jpg', '.jpeg'))
        ]
        selected_images = random.sample(images, min(len(images), subset_size_per_class))

        for img_path in selected_images:
            shutil.copy(img_path, os.path.join(subset_data_path, class_name))

    print(f"Subset created at '{subset_data_path}' with {subset_size} images.")


def get_data(config):
    """
    Create data loader from configuration.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        DataLoader object
    """
    transforms_list = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = ImageFolder(root=config["dataset_path"], transform=transforms_list)
    dataloader = DataLoader(
        dataset, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    return dataloader


def fix_state_dict(state_dict):
    """
    Adjust state dictionaries loaded from DataParallel models.
    
    Args:
        state_dict (dict): Model state dictionary
        
    Returns:
        Cleaned state dictionary
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    return new_state_dict


def plot_images(images, title="Generated Images"):
    """
    Plot a grid of images.
    
    Args:
        images: List of image tensors or a single tensor
        title (str): Title of the plot
    """
    # Stack images if list
    if isinstance(images, list):
        images = torch.stack(images)

    # Convert data type if necessary
    if images.dtype == torch.uint8:
        images = images.float() / 255.0
    elif images.dtype in [torch.float32, torch.float64]:
        images = images.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported image data type: {images.dtype}")

    # Create grid
    grid = torchvision.utils.make_grid(images, nrow=8, normalize=False, padding=2)
    np_grid = grid.permute(1, 2, 0).cpu().numpy()

    # Plot
    plt.figure(figsize=(12, 12))
    plt.title(title)
    plt.imshow(np_grid)
    plt.axis('off')
    plt.show()


def save_images(images, path):
    """
    Save a list of images to a specified directory.
    
    Args:
        images (list): List of image tensors (torch.uint8)
        path (str): Directory path to save images
    """
    os.makedirs(path, exist_ok=True)
    
    for i, image in enumerate(images):
        ndarr = image.permute(1, 2, 0).to('cpu').numpy()
        im = Image.fromarray(ndarr)
        im.save(os.path.join(path, f"{i}.png"))

    print(f"Images saved to: {path}")


def print_size_of_model(model):
    """
    Calculate and print the size of a model in MB.
    
    Args:
        model: PyTorch model
    """
    torch.save(model.state_dict(), "temp.p")
    size_mb = os.path.getsize("temp.p") / 1e6
    print(f"Model Size: {size_mb:.2f} MB")
    os.remove("temp.p")


def inspect_model(checkpoint_path, model_class, num_classes=10, compress=1, device="cuda"):
    """
    Inspect parameters of a saved model.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        model_class: Model class to instantiate
        num_classes (int): Number of classes
        compress (int): Compression factor
        device (str): Device to load model on
        
    Returns:
        Loaded model
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    model = model_class(num_classes=num_classes, compress=compress).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    state_dict = fix_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)

    # Print summary
    print("\nModel Summary:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print_size_of_model(model)

    return model
