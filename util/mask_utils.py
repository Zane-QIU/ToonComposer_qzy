"""
Mask Preprocessing Utilities for Stage 2: Localized Re-denoising

This module provides utilities for processing user-provided edit masks:
- Downsampling video masks to latent space resolution
- Applying spatio-temporal feathering (Gaussian blur)
- Preparing masks for blending during denoising
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional
import PIL.Image
from einops import rearrange


def load_mask_video(
    mask_path: Union[str, Path], 
    num_frames: int,
    height: int,
    width: int,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Load a mask video from disk (either video file or image sequence).
    
    Args:
        mask_path: Path to mask video/image file or directory of frames
        num_frames: Expected number of frames
        height: Target height
        width: Target width
        device: Device to load tensor to
        
    Returns:
        Mask tensor of shape [1, T, H, W] with values in [0, 1]
        where 1 = edit region, 0 = preserve region
    """
    mask_path = Path(mask_path)
    
    if mask_path.is_file():
        # Single image (will be repeated) or video file
        if mask_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            # Single image - repeat for all frames
            mask_img = PIL.Image.open(mask_path).convert('L')
            mask_img = mask_img.resize((width, height), PIL.Image.BILINEAR)
            mask_np = np.array(mask_img).astype(np.float32) / 255.0
            
            # Repeat for all frames: [H, W] -> [T, H, W]
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).repeat(num_frames, 1, 1)
            mask_tensor = mask_tensor.unsqueeze(0)  # [1, T, H, W]
            
        else:
            # Video file - use opencv or similar
            import cv2
            cap = cv2.VideoCapture(str(mask_path))
            frames = []
            
            for _ in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    # If video is shorter, repeat last frame
                    if len(frames) > 0:
                        frames.append(frames[-1])
                    else:
                        raise ValueError(f"Could not read frames from {mask_path}")
                else:
                    # Convert to grayscale and normalize
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
                    frames.append(gray.astype(np.float32) / 255.0)
            
            cap.release()
            
            # Stack frames: [T, H, W]
            mask_np = np.stack(frames, axis=0)
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # [1, T, H, W]
    
    elif mask_path.is_dir():
        # Directory of frames
        frame_files = sorted(mask_path.glob('*.png')) + sorted(mask_path.glob('*.jpg'))
        
        if len(frame_files) == 0:
            raise ValueError(f"No image files found in {mask_path}")
        
        frames = []
        for i in range(num_frames):
            # Loop frames if not enough
            frame_idx = i % len(frame_files)
            frame_path = frame_files[frame_idx]
            
            mask_img = PIL.Image.open(frame_path).convert('L')
            mask_img = mask_img.resize((width, height), PIL.Image.BILINEAR)
            mask_np = np.array(mask_img).astype(np.float32) / 255.0
            frames.append(mask_np)
        
        # Stack frames: [T, H, W]
        mask_np = np.stack(frames, axis=0)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # [1, T, H, W]
    
    else:
        raise ValueError(f"Mask path not found: {mask_path}")
    
    return mask_tensor.to(device)


def downsample_mask_to_latent(
    mask_video: torch.Tensor,
    latent_height: int,
    latent_width: int,
    latent_frames: int,
    vae_temporal_compression: int = 4,
    vae_spatial_compression: int = 8,
) -> torch.Tensor:
    """
    Downsample a video mask to match latent space dimensions.
    
    Args:
        mask_video: Input mask of shape [B, T, H, W] with values in [0, 1]
        latent_height: Height in latent space (H // 8)
        latent_width: Width in latent space (W // 8)
        latent_frames: Frames in latent space ((T-1) // 4 + 1)
        vae_temporal_compression: Temporal compression factor (default: 4)
        vae_spatial_compression: Spatial compression factor (default: 8)
        
    Returns:
        Downsampled mask of shape [B, T_latent, H_latent, W_latent]
    """
    B, T, H, W = mask_video.shape
    
    # Spatial downsampling
    # Reshape to [B*T, 1, H, W] for 2D interpolation
    mask_flat = mask_video.reshape(B * T, 1, H, W)
    mask_spatial = F.interpolate(
        mask_flat,
        size=(latent_height, latent_width),
        mode='bilinear',
        align_corners=False
    )
    # Reshape back to [B, T, H_latent, W_latent]
    mask_spatial = mask_spatial.reshape(B, T, latent_height, latent_width)
    
    # Temporal downsampling
    # Reshape to [B, 1, T, H_latent * W_latent] for 1D temporal interpolation
    mask_spatial_flat = mask_spatial.reshape(B, 1, T, latent_height * latent_width)
    mask_temporal = F.interpolate(
        mask_spatial_flat,
        size=(latent_frames, latent_height * latent_width),
        mode='bilinear',
        align_corners=False
    )
    # Reshape to [B, T_latent, H_latent, W_latent]
    mask_latent = mask_temporal.reshape(B, latent_frames, latent_height, latent_width)
    
    return mask_latent


def apply_gaussian_feather(
    mask: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 1.0,
    temporal_kernel_size: Optional[int] = None,
    temporal_sigma: Optional[float] = None,
) -> torch.Tensor:
    """
    Apply Gaussian blur to feather mask edges in spatial and temporal dimensions.
    
    Args:
        mask: Input mask of shape [B, T, H, W]
        kernel_size: Spatial Gaussian kernel size (must be odd)
        sigma: Spatial Gaussian sigma
        temporal_kernel_size: Temporal kernel size (if None, uses kernel_size)
        temporal_sigma: Temporal sigma (if None, uses sigma)
        
    Returns:
        Feathered mask of same shape
    """
    if temporal_kernel_size is None:
        temporal_kernel_size = kernel_size
    if temporal_sigma is None:
        temporal_sigma = sigma
    
    # Ensure kernel sizes are odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    if temporal_kernel_size % 2 == 0:
        temporal_kernel_size += 1
    
    B, T, H, W = mask.shape
    device = mask.device
    
    # 1. Spatial Gaussian blur
    # Create 2D Gaussian kernel
    kernel_range = torch.arange(kernel_size, dtype=torch.float32, device=device)
    kernel_range = kernel_range - kernel_size // 2
    
    # 2D Gaussian
    y_grid, x_grid = torch.meshgrid(kernel_range, kernel_range, indexing='ij')
    spatial_kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    spatial_kernel = spatial_kernel / spatial_kernel.sum()
    spatial_kernel = spatial_kernel.view(1, 1, kernel_size, kernel_size)
    
    # Apply spatial blur
    # Reshape to [B*T, 1, H, W]
    mask_flat = mask.reshape(B * T, 1, H, W)
    padding = kernel_size // 2
    mask_spatial = F.conv2d(mask_flat, spatial_kernel, padding=padding)
    mask_spatial = mask_spatial.reshape(B, T, H, W)
    
    # 2. Temporal Gaussian blur
    # Create 1D temporal Gaussian kernel
    temporal_range = torch.arange(temporal_kernel_size, dtype=torch.float32, device=device)
    temporal_range = temporal_range - temporal_kernel_size // 2
    temporal_kernel = torch.exp(-(temporal_range**2) / (2 * temporal_sigma**2))
    temporal_kernel = temporal_kernel / temporal_kernel.sum()
    temporal_kernel = temporal_kernel.view(1, 1, temporal_kernel_size, 1, 1)
    
    # Apply temporal blur
    # Reshape to [B, 1, T, H*W]
    mask_temporal_in = mask_spatial.reshape(B, 1, T, H * W)
    temporal_padding = temporal_kernel_size // 2
    
    # Pad temporally
    mask_temporal_padded = F.pad(
        mask_temporal_in, 
        (0, 0, temporal_padding, temporal_padding), 
        mode='replicate'
    )
    
    # Convolve along temporal dimension
    # Reshape for 3D conv: [B, 1, T_padded, H, W]
    mask_temporal_padded = mask_temporal_padded.reshape(B, 1, T + 2*temporal_padding, H, W)
    temporal_kernel_3d = temporal_kernel.squeeze(-1).squeeze(-1).view(1, 1, temporal_kernel_size, 1, 1)
    
    # Manual temporal convolution
    mask_temporal_out = torch.zeros(B, 1, T, H, W, device=device)
    for t in range(T):
        t_start = t
        t_end = t + temporal_kernel_size
        mask_temporal_out[:, :, t:t+1, :, :] = (
            mask_temporal_padded[:, :, t_start:t_end, :, :] * 
            temporal_kernel_3d
        ).sum(dim=2, keepdim=True)
    
    # Reshape back to [B, T, H, W]
    mask_feathered = mask_temporal_out.reshape(B, T, H, W)
    
    return mask_feathered


def prepare_mask_for_denoising(
    mask_video: torch.Tensor,
    latent_channels: int,
    latent_shape: Tuple[int, int, int, int],  # [B, C, T, H, W]
    feather_spatial: int = 5,
    feather_sigma: float = 2.0,
    feather_temporal: Optional[int] = None,
    feather_temporal_sigma: Optional[float] = None,
) -> torch.Tensor:
    """
    Complete preprocessing pipeline for edit masks.
    
    Args:
        mask_video: Input mask video [B, T_video, H_video, W_video]
        latent_channels: Number of channels in latent space (typically 16)
        latent_shape: Target latent shape [B, C, T_latent, H_latent, W_latent]
        feather_spatial: Spatial feathering kernel size (0 to disable)
        feather_sigma: Spatial feathering sigma
        feather_temporal: Temporal feathering kernel size (None = same as spatial)
        feather_temporal_sigma: Temporal feathering sigma (None = same as spatial)
        
    Returns:
        Prepared mask of shape [B, 1, T_latent, H_latent, W_latent]
        Ready to broadcast across channels during denoising
    """
    B, C, T_latent, H_latent, W_latent = latent_shape
    
    # Step 1: Downsample to latent resolution
    mask_latent = downsample_mask_to_latent(
        mask_video,
        latent_height=H_latent,
        latent_width=W_latent,
        latent_frames=T_latent
    )
    
    # Step 2: Apply feathering if requested
    if feather_spatial > 0:
        mask_latent = apply_gaussian_feather(
            mask_latent,
            kernel_size=feather_spatial,
            sigma=feather_sigma,
            temporal_kernel_size=feather_temporal,
            temporal_sigma=feather_temporal_sigma
        )
    
    # Step 3: Clamp to [0, 1] range
    mask_latent = mask_latent.clamp(0.0, 1.0)
    
    # Step 4: Add channel dimension for broadcasting
    # [B, T, H, W] -> [B, 1, T, H, W]
    mask_latent = mask_latent.unsqueeze(1)
    
    return mask_latent


def create_bbox_mask(
    num_frames: int,
    height: int,
    width: int,
    bbox: Tuple[int, int, int, int],  # (x, y, w, h)
    frame_range: Optional[Tuple[int, int]] = None,  # (start_frame, end_frame)
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Create a simple bounding box mask for quick testing.
    
    Args:
        num_frames: Number of frames
        height: Video height
        width: Video width
        bbox: Bounding box (x, y, w, h) in pixels
        frame_range: Optional (start_frame, end_frame) to limit temporal extent
        device: Device for tensor
        
    Returns:
        Mask tensor [1, T, H, W] with 1 inside bbox, 0 outside
    """
    mask = torch.zeros(1, num_frames, height, width, device=device)
    
    x, y, w, h = bbox
    start_frame = frame_range[0] if frame_range else 0
    end_frame = frame_range[1] if frame_range else num_frames
    
    mask[:, start_frame:end_frame, y:y+h, x:x+w] = 1.0
    
    return mask


def visualize_mask(
    mask: torch.Tensor,
    save_path: Optional[Union[str, Path]] = None,
    num_frames_to_show: int = 8
) -> None:
    """
    Visualize a mask by saving representative frames.
    
    Args:
        mask: Mask tensor [B, T, H, W] or [B, 1, T, H, W]
        save_path: Path to save visualization (creates dir if needed)
        num_frames_to_show: Number of evenly-spaced frames to visualize
    """
    # Handle both [B, T, H, W] and [B, 1, T, H, W]
    if mask.dim() == 5:
        mask = mask.squeeze(1)
    
    B, T, H, W = mask.shape
    
    # Select evenly spaced frames
    frame_indices = torch.linspace(0, T-1, num_frames_to_show).long()
    
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for idx, frame_idx in enumerate(frame_indices):
            frame = mask[0, frame_idx].cpu().numpy()
            frame_uint8 = (frame * 255).astype(np.uint8)
            img = PIL.Image.fromarray(frame_uint8, mode='L')
            img.save(save_path / f"mask_frame_{frame_idx:03d}.png")
        
        print(f"✓ Saved {len(frame_indices)} mask frames to {save_path}")
