"""
Simple UNet architecture for MNIST (28x28 grayscale images, 10 classes)
Based on EDM architecture but simplified for educational purposes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_sigmas_karras(n, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    """Generate Karras noise schedule"""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResBlock(nn.Module):
    """Residual block with time conditioning"""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class AttentionBlock(nn.Module):
    """Self-attention block"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = q.view(B, C, H * W).permute(0, 2, 1)
        k = k.view(B, C, H * W)
        v = v.view(B, C, H * W).permute(0, 2, 1)
        
        # Attention
        scale = (C // 1) ** -0.5
        attn = torch.softmax(q @ k * scale, dim=-1)
        h = (attn @ v).permute(0, 2, 1).view(B, C, H, W)
        
        return x + self.proj(h)


class SimpleUNet(nn.Module):
    """Simple UNet for MNIST"""
    def __init__(self, img_channels=1, label_dim=10, time_emb_dim=128):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_embed = TimeEmbedding(time_emb_dim)
        
        # Label embedding
        self.label_embed = nn.Embedding(label_dim, time_emb_dim)
        
        # Downsampling
        self.conv_in = nn.Conv2d(img_channels, 64, 3, padding=1)
        self.down1 = ResBlock(64, 128, time_emb_dim)
        self.down2 = ResBlock(128, 256, time_emb_dim)
        self.down3 = ResBlock(256, 512, time_emb_dim)
        
        # Middle
        self.mid_block1 = ResBlock(512, 512, time_emb_dim)
        self.mid_attn = AttentionBlock(512)
        self.mid_block2 = ResBlock(512, 512, time_emb_dim)
        
        # Upsampling
        self.up1 = ResBlock(512 + 256, 256, time_emb_dim)
        self.up2 = ResBlock(256 + 128, 128, time_emb_dim)
        self.up3 = ResBlock(128 + 64, 64, time_emb_dim)
        
        # Output
        self.norm_out = nn.GroupNorm(8, 64)
        self.conv_out = nn.Conv2d(64, img_channels, 3, padding=1)
        
    def forward(self, x, sigma, label, return_bottleneck=False):
        """
        Args:
            x: [B, C, H, W] input image/noise
            sigma: [B] noise level (can be scalar or tensor)
            label: [B] class labels (can be one-hot [B, num_classes] or class indices [B])
            return_bottleneck: if True, return bottleneck features for classifier
        """
        # Handle sigma
        if isinstance(sigma, (int, float)) or (isinstance(sigma, torch.Tensor) and sigma.dim() == 0):
            sigma = torch.full((x.shape[0],), float(sigma), device=x.device)
        elif sigma.dim() > 1:
            sigma = sigma.squeeze()
        
        # Handle label
        if label.dim() > 1:
            # One-hot to class index
            label = label.argmax(dim=1)
        
        # Time embedding from sigma
        time_emb = self.time_embed(sigma)
        label_emb = self.label_embed(label)
        time_emb = time_emb + label_emb
        
        # Downsampling
        h1 = self.conv_in(x)
        h2 = self.down1(h1, time_emb)
        h2_down = F.avg_pool2d(h2, 2)
        h3 = self.down2(h2_down, time_emb)
        h3_down = F.avg_pool2d(h3, 2)
        h4 = self.down3(h3_down, time_emb)
        h4_down = F.avg_pool2d(h4, 2)
        
        # Middle
        h = self.mid_block1(h4_down, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)
        
        if return_bottleneck:
            return h  # Return bottleneck for classifier
        
        # Upsampling
        h = F.interpolate(h, size=h3.shape[2:], mode='nearest')
        h = torch.cat([h, h3], dim=1)
        h = self.up1(h, time_emb)
        
        h = F.interpolate(h, size=h2.shape[2:], mode='nearest')
        h = torch.cat([h, h2], dim=1)
        h = self.up2(h, time_emb)
        
        h = F.interpolate(h, size=h1.shape[2:], mode='nearest')
        h = torch.cat([h, h1], dim=1)
        h = self.up3(h, time_emb)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        out = self.conv_out(h)
        
        return out

