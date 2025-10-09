"""
Latent Trajectory Caching Utilities

This module provides utilities for caching latent states during video generation
to enable iterative correction. It saves each z_t timestep to disk in fp16 format
along with metadata for reconstruction.
"""

import os
import json
import torch
import numpy as np
from typing import Dict, Any, Optional, Union
from pathlib import Path
import hashlib


class LatentTrajectoryWriter:
    """
    Writes latent trajectory (z_t at each timestep) to disk for iterative correction.
    
    Saves latents in fp16 format (.npy files) and metadata in JSON format.
    Keeps VRAM usage flat by moving tensors to CPU before writing.
    
    Args:
        cache_dir: Directory to save latent states and metadata
        use_memmap: If True, use memory-mapped arrays for large latents (default: False)
    """
    
    def __init__(self, cache_dir: str, use_memmap: bool = False):
        self.cache_dir = Path(cache_dir)
        self.use_memmap = use_memmap
        self.latents_dir = self.cache_dir / "latents"
        self.meta_path = self.cache_dir / "meta.json"
        
        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.latents_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata: Dict[str, Any] = {}
        self.timesteps_written = []
        
    def write_meta(self, metadata: Dict[str, Any]) -> None:
        """
        Write metadata about the generation process.
        
        Args:
            metadata: Dictionary containing:
                - timesteps: List of timestep values
                - num_steps: Number of inference steps
                - scheduler_type: Type of scheduler used
                - cfg_scale: Classifier-free guidance scale
                - seed: Random seed
                - latent_shape: Shape of latent tensors (C, T, H, W)
                - model_hash: Hash of model weights (optional)
                - vae_hash: Hash of VAE weights (optional)
                - prompt: Text prompt (optional)
                - negative_prompt: Negative text prompt (optional)
        """
        self.metadata.update(metadata)
        self._save_metadata()
    
    def write(self, timestep: Union[int, float, torch.Tensor], latent: torch.Tensor) -> None:
        """
        Write a latent state for a specific timestep to disk.
        
        Args:
            timestep: The timestep value (can be tensor, will be converted)
            latent: The latent tensor to save (will be moved to CPU and converted to fp16)
        """
        # Convert timestep to scalar
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.item()
        
        # Move to CPU and convert to fp16 to save VRAM
        latent_cpu = latent.detach().to('cpu', dtype=torch.float16)
        
        # Create filename with zero-padded timestep
        timestep_str = f"{int(timestep):04d}"
        filename = f"t_{timestep_str}.fp16.npy"
        filepath = self.latents_dir / filename
        
        # Convert to numpy and save
        latent_np = latent_cpu.numpy()
        
        if self.use_memmap:
            # Save as memory-mapped array
            memmap_file = np.lib.format.open_memmap(
                str(filepath), 
                mode='w+', 
                dtype=np.float16, 
                shape=latent_np.shape
            )
            memmap_file[:] = latent_np
            memmap_file.flush()
        else:
            # Save as regular numpy array
            np.save(str(filepath), latent_np)
        
        # Track written timesteps
        self.timesteps_written.append(float(timestep))
        
        # Update metadata with actual saved timesteps
        self.metadata['timesteps_written'] = sorted(self.timesteps_written, reverse=True)
        self._save_metadata()
    
    def finalize(self) -> None:
        """
        Finalize the caching process and write final metadata.
        """
        # Add summary info
        self.metadata['num_latents_saved'] = len(self.timesteps_written)
        self.metadata['cache_dir'] = str(self.cache_dir.absolute())
        self.metadata['latents_dir'] = str(self.latents_dir.absolute())
        
        self._save_metadata()
        
        print(f"✓ Latent trajectory saved to: {self.cache_dir}")
        print(f"  - Saved {len(self.timesteps_written)} latent states")
        print(f"  - Latents directory: {self.latents_dir}")
        print(f"  - Metadata: {self.meta_path}")
    
    def _save_metadata(self) -> None:
        """Internal method to save metadata to JSON file."""
        with open(self.meta_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    @staticmethod
    def compute_model_hash(model: torch.nn.Module, length: int = 8) -> str:
        """
        Compute a hash of model parameters for versioning.
        
        Args:
            model: PyTorch model
            length: Length of hash string to return
            
        Returns:
            Hash string
        """
        hasher = hashlib.sha256()
        for param in model.parameters():
            hasher.update(param.data.cpu().numpy().tobytes())
        return hasher.hexdigest()[:length]


class LatentTrajectoryReader:
    """
    Reads cached latent trajectory from disk.
    
    Args:
        cache_dir: Directory containing cached latents and metadata
    """
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.latents_dir = self.cache_dir / "latents"
        self.meta_path = self.cache_dir / "meta.json"
        
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")
        
        with open(self.meta_path, 'r') as f:
            self.metadata = json.load(f)
    
    def read(self, timestep: Union[int, float], device: str = 'cpu') -> torch.Tensor:
        """
        Read a latent state for a specific timestep.
        
        Args:
            timestep: The timestep value to load
            device: Device to load tensor to (default: 'cpu')
            
        Returns:
            Latent tensor
        """
        timestep_str = f"{int(timestep):04d}"
        filename = f"t_{timestep_str}.fp16.npy"
        filepath = self.latents_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Latent file not found: {filepath}")
        
        # Load numpy array
        latent_np = np.load(str(filepath))
        
        # Convert to torch tensor
        latent = torch.from_numpy(latent_np).to(device)
        
        return latent
    
    def read_all(self, device: str = 'cpu') -> Dict[float, torch.Tensor]:
        """
        Read all cached latent states.
        
        Args:
            device: Device to load tensors to (default: 'cpu')
            
        Returns:
            Dictionary mapping timesteps to latent tensors
        """
        latents = {}
        for timestep in self.metadata.get('timesteps_written', []):
            latents[timestep] = self.read(timestep, device=device)
        return latents
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get the cached metadata."""
        return self.metadata
    
    def get_timesteps(self) -> list:
        """Get list of available timesteps."""
        return self.metadata.get('timesteps_written', [])
