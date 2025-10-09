"""
Localized Re-denoising for Stage 2: Iterative Correction

This module implements the core editing functionality that allows re-generating
specific masked regions while preserving the rest of the video using cached latents.
"""

import torch
from typing import Dict, Any, Optional, Union
from pathlib import Path
from util.latent_cache import LatentTrajectoryReader
from util.mask_utils import prepare_mask_for_denoising


class LocalizedEditor:
    """
    Handles localized re-denoising using cached latent trajectories.
    
    This class enables editing specific regions of a generated video by:
    1. Loading cached latent states from a previous generation
    2. Re-running denoising with new guidance in masked regions
    3. Blending edited regions with cached background at each step
    """
    
    def __init__(
        self,
        cache_dir: str,
        device: str = 'cuda',
        torch_dtype: torch.dtype = torch.float16
    ):
        """
        Initialize the localized editor.
        
        Args:
            cache_dir: Directory containing cached latent trajectory
            device: Device for computation
            torch_dtype: Data type for computation
        """
        self.cache_reader = LatentTrajectoryReader(cache_dir)
        self.device = device
        self.torch_dtype = torch_dtype
        self.metadata = self.cache_reader.get_metadata()
        
        # Extract key parameters from cache metadata
        self.cached_timesteps = self.cache_reader.get_timesteps()
        self.latent_shape = self.metadata['latent_shape']
        self.original_seed = self.metadata.get('seed')
        self.cfg_scale = self.metadata.get('cfg_scale', 5.0)
        
        print(f"✓ Loaded cache from: {cache_dir}")
        print(f"  - Original prompt: {self.metadata.get('prompt', 'N/A')}")
        print(f"  - Latent shape: {self.latent_shape}")
        print(f"  - Cached timesteps: {len(self.cached_timesteps)}")
    
    def validate_compatibility(
        self,
        scheduler_timesteps: list,
        new_guidance_shape: Optional[tuple] = None
    ) -> bool:
        """
        Validate that new generation is compatible with cache.
        
        Args:
            scheduler_timesteps: Timesteps from new scheduler
            new_guidance_shape: Shape of new guidance (if applicable)
            
        Returns:
            True if compatible, raises ValueError otherwise
        """
        # Check that scheduler timesteps match cached ones
        scheduler_timesteps_list = [float(t.item() if torch.is_tensor(t) else t) 
                                     for t in scheduler_timesteps]
        
        # Note: N timesteps produce N+1 latent states (initial + N steps)
        # So cached_timesteps should have one more entry than scheduler timesteps
        expected_cache_count = len(scheduler_timesteps_list) + 1
        if len(self.cached_timesteps) != expected_cache_count:
            raise ValueError(
                f"Number of timesteps mismatch: "
                f"cache has {len(self.cached_timesteps)} latent states, "
                f"but {len(scheduler_timesteps_list)} timesteps should produce "
                f"{expected_cache_count} states (initial + {len(scheduler_timesteps_list)} steps)"
            )
        
        return True
    
    def get_initial_latent(
        self,
        start_timestep: Optional[Union[int, float]] = None,
        device: Optional[str] = None
    ) -> torch.Tensor:
        """
        Get the initial latent state to start editing from.
        
        Args:
            start_timestep: Timestep to start from (default: use largest/earliest)
            device: Device to load to (default: self.device)
            
        Returns:
            Initial latent tensor
        """
        if device is None:
            device = self.device
        
        if start_timestep is None:
            # Use the first (largest) timestep
            start_timestep = self.cached_timesteps[0]
        
        z_t = self.cache_reader.read(start_timestep, device=device)
        
        return z_t.to(dtype=self.torch_dtype)
    
    def blend_step(
        self,
        z_new: torch.Tensor,
        timestep: Union[int, float],
        mask: torch.Tensor,
        use_cached: bool = True
    ) -> torch.Tensor:
        """
        Blend newly denoised latent with cached background.
        
        Args:
            z_new: Newly denoised latent [B, C, T, H, W]
            timestep: Current timestep value
            mask: Edit mask [B, 1, T, H, W] where 1=edit, 0=preserve
            use_cached: If True, blend with cached latent; if False, return z_new
            
        Returns:
            Blended latent tensor
        """
        if not use_cached:
            return z_new
        
        # Load cached latent for this timestep
        z_cached = self.cache_reader.read(timestep, device=z_new.device)
        z_cached = z_cached.to(dtype=z_new.dtype)
        
        # Broadcast mask across channels: [B, 1, T, H, W] -> [B, C, T, H, W]
        # mask values: 1.0 = use new (edited), 0.0 = use cached (preserve)
        z_blended = mask * z_new + (1.0 - mask) * z_cached
        
        return z_blended
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get the cached metadata."""
        return self.metadata


def edit_with_cache(
    editor: LocalizedEditor,
    model,
    scheduler,
    new_prompt_emb: Dict[str, torch.Tensor],
    new_image_emb: Dict[str, torch.Tensor],
    extra_input: Dict[str, Any],
    mask_latent: torch.Tensor,
    cfg_scale: float = 5.0,
    negative_prompt_emb: Optional[Dict[str, torch.Tensor]] = None,
    start_timestep: Optional[Union[int, float]] = None,
    progress_bar=None,
    device: str = 'cuda',
    latent_writer=None,
) -> torch.Tensor:
    """
    Perform localized re-denoising using cached trajectory.
    
    This is the core editing function that:
    1. Starts from a cached latent state
    2. Runs denoising with new guidance
    3. Blends results with cached background at each step
    
    Args:
        editor: LocalizedEditor instance with cache access
        model: DIT model
        scheduler: Flow matching scheduler (must match cache)
        new_prompt_emb: New prompt embeddings
        new_image_emb: New image embeddings (if using new reference)
        extra_input: Extra inputs for model (sequence_cond, etc.)
        mask_latent: Edit mask [B, 1, T, H, W] in latent space
        cfg_scale: Classifier-free guidance scale
        negative_prompt_emb: Negative prompt embeddings
        start_timestep: Timestep to start editing from (None = start from beginning)
        progress_bar: Progress bar (e.g., tqdm)
        device: Computation device
        latent_writer: Optional LatentTrajectoryWriter to cache edited generation
        
    Returns:
        Final denoised latent z_0
    """
    # Validate compatibility
    editor.validate_compatibility(scheduler.timesteps)
    
    # Get initial latent state
    z_t = editor.get_initial_latent(start_timestep, device=device)
    
    # Cache initial state if latent_writer is provided
    if latent_writer is not None:
        if start_timestep is not None:
            latent_writer.write(start_timestep, z_t)
        else:
            latent_writer.write(scheduler.timesteps[0], z_t)
    
    # Determine which timesteps to process
    if start_timestep is not None:
        # Find index of start_timestep in scheduler
        timesteps_list = [t.item() if torch.is_tensor(t) else float(t) 
                          for t in scheduler.timesteps]
        try:
            start_idx = timesteps_list.index(float(start_timestep))
        except ValueError:
            raise ValueError(f"start_timestep {start_timestep} not found in scheduler timesteps")
        
        timesteps_to_process = scheduler.timesteps[start_idx:]
    else:
        timesteps_to_process = scheduler.timesteps
    
    # Denoising loop
    if progress_bar is not None:
        timesteps_iter = progress_bar(timesteps_to_process)
    else:
        timesteps_iter = timesteps_to_process
    
    for progress_id, timestep in enumerate(timesteps_iter):
        timestep_scalar = timestep.item() if torch.is_tensor(timestep) else float(timestep)
        timestep_tensor = timestep.unsqueeze(0).to(dtype=torch.float32, device=device) if not torch.is_tensor(timestep) else timestep.unsqueeze(0).to(dtype=torch.float32, device=device)
        
        # Model inference with new guidance
        noise_pred_posi = model(
            z_t, 
            timestep=timestep_tensor, 
            **new_prompt_emb, 
            **new_image_emb, 
            **extra_input
        )
        
        if isinstance(noise_pred_posi, tuple):
            noise_pred_posi = noise_pred_posi[0]
        
        # CFG if enabled
        if cfg_scale != 1.0 and negative_prompt_emb is not None:
            noise_pred_nega = model(
                z_t, 
                timestep=timestep_tensor, 
                **negative_prompt_emb, 
                **new_image_emb, 
                **extra_input
            )
            if isinstance(noise_pred_nega, tuple):
                noise_pred_nega = noise_pred_nega[0]
            
            noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
        else:
            noise_pred = noise_pred_posi
        
        # Scheduler step to get next latent
        z_next_new = scheduler.step(
            noise_pred, 
            scheduler.timesteps[start_idx + progress_id] if start_timestep else scheduler.timesteps[progress_id], 
            z_t
        )
        
        # Determine next timestep for blending
        if start_timestep:
            actual_progress_id = start_idx + progress_id
        else:
            actual_progress_id = progress_id
            
        if actual_progress_id + 1 < len(scheduler.timesteps):
            next_timestep = scheduler.timesteps[actual_progress_id + 1]
            next_timestep_val = next_timestep.item() if torch.is_tensor(next_timestep) else float(next_timestep)
        else:
            # Final step
            next_timestep_val = 0
        
        # Blend with cached trajectory
        z_t = editor.blend_step(
            z_new=z_next_new,
            timestep=next_timestep_val,
            mask=mask_latent,
            use_cached=True
        )
        
        # Cache the edited latent state if latent_writer is provided
        if latent_writer is not None:
            latent_writer.write(next_timestep_val, z_t)
    
    return z_t


def create_edit_mask_from_regions(
    num_frames: int,
    height: int,
    width: int,
    regions: list,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Create an edit mask from a list of region specifications.
    
    Args:
        num_frames: Number of frames
        height: Video height
        width: Video width
        regions: List of region dicts, each with:
            - 'bbox': (x, y, w, h) in pixels
            - 'frames': (start_frame, end_frame) or None for all frames
            - 'value': mask value (default 1.0)
        device: Device for tensor
        
    Returns:
        Combined mask tensor [1, T, H, W]
    """
    mask = torch.zeros(1, num_frames, height, width, device=device)
    
    for region in regions:
        x, y, w, h = region['bbox']
        start_frame, end_frame = region.get('frames', (0, num_frames))
        value = region.get('value', 1.0)
        
        mask[:, start_frame:end_frame, y:y+h, x:x+w] = value
    
    return mask
