"""
Model architectures for compact diffusion models.
Contains UNet, Self-Attention, and EMA implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Multi-head self-attention layer with feed-forward network.
    
    Args:
        channels (int): Number of input/output channels
        size (int): Spatial size of the feature map
    """
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        """
        Forward pass of self-attention.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        channels = x.shape[1]
        # Reshape: (B, C, H, W) -> (B, H*W, C)
        x = x.view(-1, channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        return attention_value.swapaxes(2, 1).view(-1, channels, self.size, self.size)


class DoubleConv(nn.Module):
    """
    Double convolution block with optional residual connection.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        mid_channels (int, optional): Number of intermediate channels
        residual (bool): Whether to use residual connection
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """
    Downsampling block with time embedding.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        emb_dim (int): Dimension of time embedding
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        """
        Args:
            x: Input tensor
            t: Time embedding
        """
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    """
    Upsampling block with skip connections and time embedding.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        emb_dim (int): Dimension of time embedding
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        """
        Args:
            x: Input tensor from previous layer
            skip_x: Skip connection from encoder
            t: Time embedding
        """
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet_conditional(nn.Module):
    """
    Conditional U-Net for diffusion models with time and class conditioning.
    
    Args:
        c_in (int): Number of input channels
        c_out (int): Number of output channels
        time_dim (int): Dimension of time embedding
        num_classes (int, optional): Number of classes for conditional generation
        compress (int): Compression factor for model size
        device (str): Device to run the model on
    """
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, compress=1, device="cuda"):
        super().__init__()
        c_0 = 64 // compress
        self.device = device
        self.time_dim = time_dim
        
        # Encoder
        self.inc = DoubleConv(c_in, c_0)
        self.down1 = Down(c_0, c_0*2)
        self.sa1 = SelfAttention(c_0*2, 32)
        self.down2 = Down(c_0*2, c_0*4)
        self.sa2 = SelfAttention(c_0*4, 16)
        self.down3 = Down(c_0*4, c_0*4)
        self.sa3 = SelfAttention(c_0*4, 8)

        # Bottleneck
        self.bot1 = DoubleConv(c_0*4, c_0*8)
        self.bot2 = DoubleConv(c_0*8, c_0*8)
        self.bot3 = DoubleConv(c_0*8, c_0*4)

        # Decoder
        self.up1 = Up(c_0*8, c_0*2)
        self.sa4 = SelfAttention(c_0*2, 16)
        self.up2 = Up(c_0*4, c_0)
        self.sa5 = SelfAttention(c_0, 32)
        self.up3 = Up(c_0*2, c_0)
        self.sa6 = SelfAttention(c_0, 64)
        self.outc = nn.Conv2d(c_0, c_out, kernel_size=1)

        # Class conditioning
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        """
        Sinusoidal position encoding for timesteps.
        
        Args:
            t: Timestep tensor
            channels: Number of channels for encoding
        Returns:
            Position encoding tensor
        """
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2, device=t.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y=None):
        """
        Forward pass of the U-Net.
        
        Args:
            x: Input image tensor
            t: Timestep tensor
            y: Class labels (optional)
        Returns:
            Predicted noise
        """
        # Time embedding
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        # Add class embedding if provided
        if y is not None and hasattr(self, 'label_emb'):
            if y.size(0) != x.size(0):
                y = y[:x.size(0)]
            t = t + self.label_emb(y)

        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        # Bottleneck
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        # Decoder path
        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


class EMA:
    """
    Exponential Moving Average for model parameters.
    
    Args:
        beta (float): Decay rate for moving average
    """
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        """Update the moving average of model parameters."""
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        """Compute exponential moving average."""
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        """
        Update EMA model.
        
        Args:
            ema_model: EMA model to update
            model: Current training model
            step_start_ema: Step to start EMA updates
        """
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        """Reset EMA model parameters to current model."""
        ema_model.load_state_dict(model.state_dict())
