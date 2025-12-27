"""
Diffusion process for training and sampling.
Implements the forward and reverse diffusion processes.
"""

import torch
import torch.nn as nn
import logging
from tqdm import tqdm


class Diffusion:
    """
    Implements the diffusion process for training and sampling.
    
    Args:
        noise_steps (int): Number of diffusion timesteps
        beta_start (float): Starting value of noise schedule
        beta_end (float): Ending value of noise schedule
        img_size (int): Size of images (assuming square images)
        device (str): Device to run computations on
    """
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        # Prepare noise schedule
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        """
        Prepare linear noise schedule.
        
        Returns:
            Tensor of beta values for each timestep
        """
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        """
        Add noise to images according to the diffusion process.
        
        Args:
            x: Clean images of shape (B, C, H, W)
            t: Timesteps for each image in the batch
            
        Returns:
            Tuple of (noisy_images, noise)
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, n):
        """
        Sample random timesteps.
        
        Args:
            n: Number of timesteps to sample
            
        Returns:
            Tensor of random timesteps
        """
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, total_images, batch_size, labels=None, cfg_scale=0):
        """
        Generate images using the trained model.
        
        Args:
            model: Trained diffusion model
            total_images: Total number of images to generate
            batch_size: Batch size for generation
            labels: Class labels for conditional generation (optional)
            cfg_scale: Classifier-free guidance scale
            
        Returns:
            List of generated images as uint8 tensors
        """
        logging.info(f"Sampling {total_images} new images...")
        
        # Wrap model in DataParallel if not already
        if not isinstance(model, nn.DataParallel):
            model = nn.DataParallel(model)
        
        model.to(self.device)
        model.eval()

        generated_images = []
        
        with torch.no_grad():
            for batch_idx in tqdm(range(0, total_images, batch_size), desc="Generating images"):
                current_batch_size = min(batch_size, total_images - len(generated_images))
                
                # Start from random noise
                x = torch.randn((current_batch_size, 3, self.img_size, self.img_size)).to(self.device)

                # Reverse diffusion process
                for i in tqdm(reversed(range(1, self.noise_steps)), position=0, leave=False, desc="Denoising"):
                    t = (torch.ones(current_batch_size) * i).long().to(self.device)
                    
                    # Classifier-Free Guidance
                    if cfg_scale > 0 and labels is not None:
                        # Generate unconditional predictions
                        uncond_predicted_noise = model(x, t, y=None)
                        # Generate conditional predictions
                        cond_predicted_noise = model(x, t, y=labels[:current_batch_size])
                        # Combine predictions
                        predicted_noise = torch.lerp(uncond_predicted_noise, cond_predicted_noise, cfg_scale)
                    else:
                        # Standard prediction
                        y_batch = labels[:current_batch_size] if labels is not None else None
                        predicted_noise = model(x, t, y=y_batch)

                    # Reverse step
                    alpha = self.alpha[t][:, None, None, None]
                    alpha_hat = self.alpha_hat[t][:, None, None, None]
                    beta = self.beta[t][:, None, None, None]

                    if i > 1:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.zeros_like(x)

                    x = (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) / torch.sqrt(alpha) + torch.sqrt(beta) * noise

                # Post-processing: convert to [0, 255] uint8
                x = (x.clamp(-1, 1) + 1) / 2
                x = (x * 255).type(torch.uint8)
                generated_images.extend(x.cpu())

        model.train()
        return generated_images
