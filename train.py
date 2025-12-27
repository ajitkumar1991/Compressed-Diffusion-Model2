"""
Training script for compact diffusion models with knowledge distillation.
"""

import os
import copy
import random
import logging
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt

from models import UNet_conditional, EMA
from diffusion import Diffusion
from utils import get_data, fix_state_dict


def train(config):
    """
    Train a UNet_conditional model based on the provided configurations.
    Supports knowledge distillation, mixed precision, EMA, and checkpointing.
    
    Args:
        config (dict): Configuration dictionary with training parameters
    """
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s: %(message)s",
        level=logging.INFO, 
        datefmt="%I:%M:%S"
    )

    # Extract configuration
    device = config.get("device", "cuda")
    batch_size = config.get("batch_size", 64)
    epochs = config.get("epochs", 300)
    lr = config.get("learning_rate", 3e-4)
    num_classes = config.get("num_classes", 10)
    image_size = config.get("image_size", 64)
    run_name = config.get("run_name")
    use_distillation = config.get("distillation", False)
    use_ema = config.get("use_ema", False)
    use_mixed_precision = config.get("mixed_precision", False)
    
    # Create directories
    checkpoint_dir = os.path.join("checkpoints", run_name)
    model_dir = os.path.join("models", run_name)
    results_dir = os.path.join("results", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Initialize data loader
    dataloader = get_data(config)

    # Initialize model
    compress_factor = 2 if config.get("compress", False) else 1
    model = UNet_conditional(
        compress=compress_factor, 
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # Initialize teacher model for distillation
    teacher = None
    if use_distillation:
        teacher_path = config.get("teacher_path", "")
        if not os.path.exists(teacher_path):
            raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")
        
        teacher = UNet_conditional(compress=1, num_classes=num_classes).to(device)
        teacher_checkpoint = torch.load(teacher_path, map_location=device)
        
        # Extract state dict if nested
        if isinstance(teacher_checkpoint, dict) and 'model_state_dict' in teacher_checkpoint:
            teacher_checkpoint = teacher_checkpoint['model_state_dict']
        
        teacher_checkpoint = fix_state_dict(teacher_checkpoint)
        teacher.load_state_dict(teacher_checkpoint, strict=True)
        teacher = torch.nn.DataParallel(teacher)
        teacher.eval()
        
        # Freeze teacher parameters
        for param in teacher.parameters():
            param.requires_grad = False
        
        logging.info(f"Loaded teacher model from: {teacher_path}")

    # Initialize optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    # Initialize diffusion
    diffusion = Diffusion(img_size=image_size, device=device)

    # Initialize EMA
    ema_model = None
    ema = None
    if use_ema:
        ema = EMA(beta=0.995)
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)
        logging.info("Using EMA")

    # Setup mixed precision
    scaler = GradScaler() if use_mixed_precision else None
    logging.info(f"Mixed precision: {use_mixed_precision}")

    # Initialize history trackers
    loss_history = []
    
    # Check for existing checkpoints to resume
    latest_epoch = 0
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.pt')[0]))
            latest_epoch = int(latest_checkpoint.split('_')[-1].split('.pt')[0])
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if scaler and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            if 'loss_history' in checkpoint:
                loss_history = checkpoint['loss_history']
            
            logging.info(f"Resumed from checkpoint: {checkpoint_path} (epoch {latest_epoch})")

    # Training loop
    for epoch in range(latest_epoch, epochs):
        logging.info(f"Starting epoch {epoch+1}/{epochs}")
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        epoch_loss = 0

        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)

            optimizer.zero_grad()

            # Mixed precision training
            if use_mixed_precision:
                with autocast():
                    x_t, noise = diffusion.noise_images(images, t)
                    
                    # Classifier-free guidance: randomly drop labels
                    labels_input = None if random.random() < 0.1 else labels
                    
                    predicted_noise = model(x_t, t, labels_input)
                    loss_simple = mse(noise, predicted_noise)
                    
                    # Knowledge distillation loss
                    if teacher is not None:
                        with torch.no_grad():
                            teacher_predicted_noise = teacher(x_t, t, labels_input)
                        loss_distill = mse(teacher_predicted_noise, predicted_noise)
                        loss = (loss_simple + loss_distill) / 2
                    else:
                        loss = loss_simple
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training
                x_t, noise = diffusion.noise_images(images, t)
                labels_input = None if random.random() < 0.1 else labels
                
                predicted_noise = model(x_t, t, labels_input)
                loss_simple = mse(noise, predicted_noise)
                
                if teacher is not None:
                    with torch.no_grad():
                        teacher_predicted_noise = teacher(x_t, t, labels_input)
                    loss_distill = mse(teacher_predicted_noise, predicted_noise)
                    loss = (loss_simple + loss_distill) / 2
                else:
                    loss = loss_simple
                
                loss.backward()
                optimizer.step()

            # Update EMA
            if ema is not None:
                ema.step_ema(ema_model, model)

            epoch_loss += loss.item()
            pbar.set_postfix(MSE=loss.item())

        # Record average epoch loss
        avg_epoch_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_epoch_loss)
        logging.info(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.6f}")

        # Save checkpoint
        if (epoch + 1) % config.get("save_interval", 10) == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"ckpt_epoch_{epoch+1}.pt")
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_history': loss_history
            }
            
            if scaler:
                checkpoint_data['scaler_state_dict'] = scaler.state_dict()
            
            torch.save(checkpoint_data, checkpoint_path)
            logging.info(f"Checkpoint saved: {checkpoint_path}")

            # Save EMA checkpoint
            if ema_model is not None:
                ema_path = os.path.join(checkpoint_dir, f"ema_ckpt_epoch_{epoch+1}.pt")
                torch.save(ema_model.state_dict(), ema_path)

    # Save final checkpoint
    final_checkpoint_path = os.path.join(model_dir, "final_model.pt")
    torch.save(model.module.state_dict(), final_checkpoint_path)
    logging.info(f"Final model saved: {final_checkpoint_path}")

    # Plot and save loss history
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss History')
    plt.legend()
    plt.savefig(os.path.join(results_dir, "loss_history.png"))
    plt.close()

    logging.info("Training completed!")
