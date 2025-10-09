#!/usr/bin/env python3
"""
ToonComposer CLI Tool
A command-line interface for ToonComposer that supports all Gradio app functionality.

Usage examples:
  # Basic generation with one image and one sketch
  python cli_tool.py --prompt "A man underwater" --image path/to/image.png --image-frame 0 \
                     --sketch path/to/sketch.png --sketch-frame 30 --num-frames 61

  # Multiple sketches with masks
  python cli_tool.py --prompt "Growing flower" --image img.png --image-frame 0 \
                     --sketch sketch1.png --sketch-frame 30 --sketch-mask mask1.png \
                     --sketch sketch2.png --sketch-frame 60 --sketch-mask mask2.png \
                     --num-frames 61 --resolution 608p

  # Load a sample
  python cli_tool.py --load-sample 1 --output output.mp4
"""

import torch
import numpy as np
from PIL import Image
import argparse
import json
import os
import tempfile
import cv2
from einops import rearrange
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from huggingface_hub import snapshot_download

from tooncomposer import ToonComposer, get_base_model_paths
from util.latent_cache import LatentTrajectoryWriter

# Reuse helper functions from app.py
WAN_REPO_ID = "Wan-AI/Wan2.1-I2V-14B-480P"
TOONCOMPOSER_REPO_ID = "TencentARC/ToonComposer"

def _path_is_dir_with_files(dir_path: str, required_files: List[str]) -> bool:
    if not dir_path or not os.path.isdir(dir_path):
        return False
    for f in required_files:
        if not os.path.exists(os.path.join(dir_path, f)):
            return False
    return True

def resolve_wan_model_root(preferred_dir: Optional[str] = None, hf_token: Optional[str] = None) -> str:
    expected = get_base_model_paths("Wan2.1-I2V-14B-480P", format='dict', model_root=".")
    required_files = []
    required_files.extend([os.path.basename(p) for p in expected["dit"]])
    required_files.append(os.path.basename(expected["image_encoder"]))
    required_files.append(os.path.basename(expected["text_encoder"]))
    required_files.append(os.path.basename(expected["vae"]))

    if _path_is_dir_with_files(preferred_dir or "", required_files):
        return os.path.abspath(preferred_dir)

    env_dir = os.environ.get("WAN21_I2V_DIR")
    if _path_is_dir_with_files(env_dir or "", required_files):
        return os.path.abspath(env_dir)

    try:
        cached_dir = snapshot_download(repo_id=WAN_REPO_ID, local_files_only=True)
        return cached_dir
    except Exception:
        pass

    cached_dir = snapshot_download(repo_id=WAN_REPO_ID, token=hf_token)
    return cached_dir

def resolve_tooncomposer_repo_dir(preferred_dir: Optional[str] = None, hf_token: Optional[str] = None) -> str:
    def has_resolution_dirs(base_dir: str) -> bool:
        if not base_dir or not os.path.isdir(base_dir):
            return False
        ok = False
        for res in ["480p", "608p"]:
            d = os.path.join(base_dir, res)
            if os.path.isdir(d):
                ckpt = os.path.join(d, "tooncomposer.ckpt")
                cfg = os.path.join(d, "config.json")
                if os.path.exists(ckpt) and os.path.exists(cfg):
                    ok = True
        return ok

    if has_resolution_dirs(preferred_dir or ""):
        return os.path.abspath(preferred_dir)

    env_dir = os.environ.get("TOONCOMPOSER_DIR")
    if has_resolution_dirs(env_dir or ""):
        return os.path.abspath(env_dir)

    try:
        cached_dir = snapshot_download(repo_id=TOONCOMPOSER_REPO_ID, local_files_only=True)
        return cached_dir
    except Exception:
        pass

    cached_dir = snapshot_download(repo_id=TOONCOMPOSER_REPO_ID, token=hf_token)
    return cached_dir

def build_checkpoints_by_resolution(tooncomposer_base_dir: str) -> Dict[str, Dict[str, object]]:
    mapping = {}
    res_to_hw = {
        "480p": (480, 832),
        "608p": (608, 1088),
    }
    for res, (h, w) in res_to_hw.items():
        res_dir = os.path.join(tooncomposer_base_dir, res)
        mapping[res] = {
            "target_height": h,
            "target_width": w,
            "snapshot_args_path": os.path.join(res_dir, "config.json"),
            "checkpoint_path": os.path.join(res_dir, "tooncomposer.ckpt"),
        }
    return mapping

def _load_model_config(config_path: str) -> Dict[str, object]:
    with open(config_path, "r") as f:
        data = json.load(f)
    return data

def _merge_with_defaults(cfg: Dict[str, object]) -> Dict[str, object]:
    defaults = {
        "base_model_name": "Wan2.1-I2V-14B-480P",
        "learning_rate": 1e-5,
        "train_architecture": None,
        "lora_rank": 4,
        "lora_alpha": 4,
        "lora_target_modules": "",
        "init_lora_weights": "kaiming",
        "use_gradient_checkpointing": True,
        "tiled": False,
        "tile_size_height": 34,
        "tile_size_width": 34,
        "tile_stride_height": 18,
        "tile_stride_width": 16,
        "output_path": "./",
        "use_dera": False,
        "dera_rank": None,
        "use_dera_spatial": True,
        "use_dera_temporal": True,
        "use_sequence_cond": True,
        "sequence_cond_mode": "sparse",
        "use_channel_cond": False,
        "use_sequence_cond_position_aware_residual": True,
        "use_sequence_cond_loss": False,
        "fast_dev": False,
        "max_num_cond_images": 1,
        "max_num_cond_sketches": 2,
        "random_spaced_cond_frames": False,
        "use_sketch_mask": True,
        "sketch_mask_ratio": 0.2,
        "no_first_sketch": False,
    }
    merged = defaults.copy()
    merged.update(cfg)
    return merged

def initialize_model(resolution="480p", fast_dev=False, device="cuda:0", dtype=torch.bfloat16,
                     wan_model_dir: Optional[str] = None, tooncomposer_dir: Optional[str] = None,
                     hf_token: Optional[str] = None, checkpoints_by_resolution=None):
    if resolution not in checkpoints_by_resolution:
        raise ValueError(f"Resolution '{resolution}' is not available. Found: {list(checkpoints_by_resolution.keys())}")

    snapshot_args_path = checkpoints_by_resolution[resolution]["snapshot_args_path"]
    checkpoint_path = checkpoints_by_resolution[resolution]["checkpoint_path"]

    snapshot_args_raw = _load_model_config(snapshot_args_path)
    snapshot_args = _merge_with_defaults(snapshot_args_raw)
    snapshot_args["checkpoint_path"] = checkpoint_path
    snapshot_args["model_root"] = resolve_wan_model_root(preferred_dir=wan_model_dir, hf_token=hf_token)

    if "training_max_frame_stride" not in snapshot_args:
        snapshot_args["training_max_frame_stride"] = 4
    snapshot_args["random_spaced_cond_frames"] = False
    args = argparse.Namespace(**snapshot_args)
    
    if not fast_dev:
        model = ToonComposer(
            base_model_name=args.base_model_name,
            model_root=args.model_root,
            learning_rate=args.learning_rate,
            use_gradient_checkpointing=args.use_gradient_checkpointing,
            checkpoint_path=args.checkpoint_path,
            tiled=args.tiled,
            tile_size=(args.tile_size_height, args.tile_size_width),
            tile_stride=(args.tile_stride_height, args.tile_stride_width),
            output_path=args.output_path,
            use_dera=args.use_dera,
            dera_rank=args.dera_rank,
            use_dera_spatial=args.use_dera_spatial,
            use_dera_temporal=args.use_dera_temporal,
            use_sequence_cond=args.use_sequence_cond,
            sequence_cond_mode=args.sequence_cond_mode,
            use_channel_cond=args.use_channel_cond,
            use_sequence_cond_position_aware_residual=args.use_sequence_cond_position_aware_residual,
            use_sequence_cond_loss=args.use_sequence_cond_loss,
            fast_dev=args.fast_dev,
            max_num_cond_images=args.max_num_cond_images,
            max_num_cond_sketches=args.max_num_cond_sketches,
            random_spaced_cond_frames=args.random_spaced_cond_frames,
            use_sketch_mask=args.use_sketch_mask,
            sketch_mask_ratio=args.sketch_mask_ratio,
            no_first_sketch=args.no_first_sketch,
        )
        model = model.to(device, dtype=dtype).eval()
    else:
        print("Fast dev mode. Models will not be loaded.")
        model = None
    print("Models initialized.")
    return model, device, dtype

def process_conditions(num_items, item_inputs, num_frames, is_sketch=False, target_height=480, target_width=832, device="cuda:0"):
    video = torch.zeros((1, 3, num_frames, target_height, target_width), device=device)
    mask = torch.zeros((1, num_frames), device=device)
    
    for i in range(num_items):
        img, frame_idx = item_inputs[i]
        if img is None or frame_idx is None:
            continue
            
        img_tensor = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 127.5 - 1.0
        if is_sketch:
            img_tensor = -img_tensor
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        _, _, h, w = img_tensor.shape
        
        if h/w < target_height/target_width:
            new_h = target_height
            new_w = int(w * (new_h / h))
        else:
            new_w = target_width
            new_h = int(h * (new_w / w))
            
        img_tensor = torch.nn.functional.interpolate(img_tensor, size=(new_h, new_w), mode="bilinear")
        
        if new_h > target_height or new_w > target_width:
            start_h = max(0, (new_h - target_height) // 2)
            start_w = max(0, (new_w - target_width) // 2)
            img_tensor = img_tensor[:, :, start_h:start_h+target_height, start_w:start_w+target_width]
        
        frame_idx = min(max(int(frame_idx), 0), num_frames-1)
        if is_sketch:
            video[:, :, frame_idx] = img_tensor[:, :3]
        else:
            video[:, :, frame_idx] = img_tensor
        mask[:, frame_idx] = 1.0
    return video, mask

def process_sketch_masks(sketch_masks_data, num_frames, target_height=480, target_width=832, device="cuda:0"):
    sketch_local_mask = torch.ones((1, 1, num_frames, target_height, target_width), device=device)
    
    for mask_img, frame_idx in sketch_masks_data:
        if mask_img is None or frame_idx is None:
            continue
            
        mask_array = np.array(mask_img)
        if len(mask_array.shape) == 3:
            mask_array = np.max(mask_array, axis=2)
        
        mask_tensor = torch.from_numpy(mask_array).float()
        if mask_tensor.max() > 1.0:
            mask_tensor = mask_tensor / 255.0
        
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        mask_tensor = torch.nn.functional.interpolate(mask_tensor, size=(target_height, target_width), mode="nearest")
        
        mask_tensor = 1.0 - mask_tensor
        
        frame_idx = min(max(int(frame_idx), 0), num_frames-1)
        sketch_local_mask[:, :, frame_idx] = mask_tensor
        
    return sketch_local_mask

def load_sample_config(sample_id: int) -> Optional[Dict]:
    """Load sample configuration"""
    sample_configs = {
        1: {
            "prompt": "Underwater scene: A shirtless man plays with a spiraling blue fish. A whale follows a bag in the man's hand, swimming in circles as the man uses the bag to lure the blue fish forward. Anime. High quality.",
            "num_sketches": 3,
            "image": "samples/1_image1.png",
            "image_frame": 0,
            "sketches": ["samples/1_sketch1.jpg", "samples/1_sketch2.jpg", "samples/1_sketch3.jpg"],
            "sketch_frames": [20, 40, 60],
            "num_frames": 61
        },
        2: {
            "prompt": "A girl and a silver-haired boy plant a huge flower. As the camera slowly moves up, the huge flower continues to grow and bloom. Anime. High quality.",
            "num_sketches": 2,
            "image": "samples/2_image1.jpg",
            "image_frame": 0,
            "sketches": ["samples/2_sketch1.jpg", "samples/2_sketch2.jpg"],
            "sketch_frames": [30, 60],
            "num_frames": 61
        },
        3: {
            "prompt": "An ancient Chinese boy holds an apple and smiles as he gives it to an elderly man nearby. Anime. High quality.",
            "num_sketches": 1,
            "image": "samples/3_image1.png",
            "image_frame": 0,
            "sketches": ["samples/3_sketch1.jpg"],
            "sketch_frames": [30],
            "num_frames": 33
        }
    }
    
    return sample_configs.get(sample_id)


def edit_video_from_cache(args, model, device, dtype, checkpoints_by_resolution):
    """
    Edit a video using cached latent trajectory (Stage 2).
    Reuses the same parameters as normal generation: sketch, image, prompt, etc.
    """
    if not args.edit_mask:
        raise ValueError("--edit_mask is required when using --edit_from_cache")
    
    print(f"=== Stage 2: Editing from cache ===")
    print(f"Cache directory: {args.edit_from_cache}")
    print(f"Edit mask: {args.edit_mask}")
    
    # Get resolution from cache or args
    from util.latent_cache import LatentTrajectoryReader
    cache_reader = LatentTrajectoryReader(args.edit_from_cache)
    cache_metadata = cache_reader.get_metadata()
    
    # Use cached resolution if not overridden
    if args.resolution == "480p":  # default value
        # Try to infer from cache
        cache_height = cache_metadata.get('height')
        if cache_height == 480:
            args.resolution = "480p"
        elif cache_height == 608:
            args.resolution = "608p"
    
    target_height = checkpoints_by_resolution[args.resolution]["target_height"]
    target_width = checkpoints_by_resolution[args.resolution]["target_width"]
    num_frames = cache_metadata.get('num_frames', args.num_frames)
    
    print(f"Resolution: {args.resolution} ({target_height}x{target_width})")
    print(f"Frames: {num_frames}")
    
    # Update model resolution
    if model is not None:
        model.update_height_width(target_height, target_width)
    
    # Use cached prompt if new one not provided
    edit_prompt = args.prompt if args.prompt else cache_metadata.get('prompt', '')
    print(f"Prompt: {edit_prompt}")
    
    # Process images (keyframes)
    masked_cond_image = None
    preserved_image_mask = None
    if args.image and args.image_frame:
        cond_images = []
        for img_path, frame_idx in zip(args.image, args.image_frame):
            img = Image.open(img_path).convert("RGB")
            cond_images.append((img, frame_idx))
        print(f"✓ Loaded {len(cond_images)} keyframe image(s)")
        
        masked_cond_image, preserved_image_mask = process_conditions(
            len(cond_images), cond_images, num_frames, is_sketch=False,
            target_height=target_height, target_width=target_width, device=device
        )
        masked_cond_image = masked_cond_image.to(device=device, dtype=dtype)
        preserved_image_mask = preserved_image_mask.to(device=device, dtype=dtype)
    
    # Process sketches
    masked_cond_sketch = None
    preserved_sketch_mask = None
    sketch_local_mask = None
    
    if args.sketch and args.sketch_frame:
        print(f"Processing {len(args.sketch)} sketch(es)")
        
        cond_sketches = []
        for sketch_path, frame_idx in zip(args.sketch, args.sketch_frame):
            sketch_img = Image.open(sketch_path).convert("RGB")
            cond_sketches.append((sketch_img, frame_idx))
        
        masked_cond_sketch, preserved_sketch_mask = process_conditions(
            len(cond_sketches), cond_sketches, num_frames, is_sketch=True,
            target_height=target_height, target_width=target_width, device=device
        )
        
        # Process sketch masks if provided
        if args.sketch_mask:
            if len(args.sketch_mask) != len(args.sketch):
                raise ValueError("Number of sketch masks must match number of sketches")
            
            sketch_masks_data = []
            for mask_path, frame_idx in zip(args.sketch_mask, args.sketch_frame):
                mask_img = Image.open(mask_path).convert("L")
                sketch_masks_data.append((mask_img, frame_idx))
            
            sketch_local_mask = process_sketch_masks(
                sketch_masks_data, num_frames,
                target_height=target_height, target_width=target_width, device=device
            )
        else:
            sketch_local_mask = torch.ones((1, 1, num_frames, target_height, target_width), device=device)
        
        masked_cond_sketch = masked_cond_sketch.to(device=device, dtype=dtype)
        preserved_sketch_mask = preserved_sketch_mask.to(device=device, dtype=dtype)
    
    if args.fast_dev:
        print("Fast dev mode, creating dummy video")
        output_path = args.output or "output_edit_dummy.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (target_width, target_height))
        for i in range(30):
            frame = np.full((target_height, target_width, 3), (i * 8) % 255, dtype=np.uint8)
            video_writer.write(frame)
        video_writer.release()
        print(f"✓ Dummy edited video saved to {output_path}")
        return output_path
    
    # Set up latent caching for this edited generation
    latent_writer = None
    if args.cache_latents:
        cache_dir = args.cache_dir or f"edited_latents_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        latent_writer = LatentTrajectoryWriter(cache_dir, use_memmap=args.use_memmap)
        print(f"✓ Caching latents to: {cache_dir}")
    
    print("Running localized editing...")
    with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(device).type):
        model.pipe.device = device
        
        edited_video = model.pipe.edit_from_cache(
            cache_dir=args.edit_from_cache,
            mask_video_path=args.edit_mask,
            prompt=edit_prompt,
            negative_prompt=model.negative_prompt,
            cfg_scale=args.cfg_scale if args.cfg_scale != 7.5 else None,  # Use cache default if not specified
            start_timestep=args.edit_start_timestep,
            feather_kernel_size=args.edit_feather_size,
            feather_sigma=args.edit_feather_sigma,
            tiled=True,
            input_condition_video=masked_cond_image,
            input_condition_preserved_mask=preserved_image_mask,
            input_condition_video_sketch=masked_cond_sketch,
            input_condition_preserved_mask_sketch=preserved_sketch_mask,
            sketch_local_mask=sketch_local_mask,
            sequence_cond_residual_scale=args.sequence_cond_residual_scale,
            latent_writer=latent_writer,
        )
    
    # Convert to video
    video_frames = model.pipe.tensor2video(edited_video[0].cpu())
    
    output_path = args.output or f"edited_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    width, height = video_frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
    
    for frame in video_frames:
        frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
    
    video_writer.release()
    print(f"✓ Edited video saved to {output_path}")
    
    return output_path


def generate_video(args, model, device, dtype, checkpoints_by_resolution):
    """Main video generation function"""
    
    # STAGE 2: Edit from cache
    if args.edit_from_cache:
        return edit_video_from_cache(args, model, device, dtype, checkpoints_by_resolution)
    
    # Load sample if specified
    if args.load_sample:
        sample_config = load_sample_config(args.load_sample)
        if not sample_config:
            raise ValueError(f"Sample {args.load_sample} not found")
        
        # Override args with sample config
        args.prompt = sample_config["prompt"]
        args.num_frames = sample_config["num_frames"]
        args.image = [sample_config["image"]]
        args.image_frame = [sample_config["image_frame"]]
        args.sketch = sample_config["sketches"]
        args.sketch_frame = sample_config["sketch_frames"]
        print(f"✓ Loaded sample {args.load_sample}")
    
    # Validate inputs
    if not args.prompt:
        raise ValueError("Text prompt is required")
    if not args.image or not args.image_frame:
        raise ValueError("At least one image with frame index is required")
    if len(args.image) != len(args.image_frame):
        raise ValueError("Number of images must match number of image frame indices")
    if args.sketch and args.sketch_frame and len(args.sketch) != len(args.sketch_frame):
        raise ValueError("Number of sketches must match number of sketch frame indices")
    
    # Get target resolution
    target_height = checkpoints_by_resolution[args.resolution]["target_height"]
    target_width = checkpoints_by_resolution[args.resolution]["target_width"]
    
    print(f"Generating {args.num_frames} frames at {args.resolution} ({target_height}x{target_width})")
    print(f"Prompt: {args.prompt}")
    
    # Update model resolution
    if model is not None:
        model.update_height_width(target_height, target_width)
    
    # Load and process images
    cond_images = []
    for img_path, frame_idx in zip(args.image, args.image_frame):
        img = Image.open(img_path).convert("RGB")
        cond_images.append((img, frame_idx))
    print(f"✓ Loaded {len(cond_images)} keyframe image(s)")
    
    # Load and process sketches
    cond_sketches = []
    sketch_masks_data = []
    if args.sketch and args.sketch_frame:
        for sketch_path, frame_idx in zip(args.sketch, args.sketch_frame):
            sketch_img = Image.open(sketch_path).convert("RGB")
            cond_sketches.append((sketch_img, frame_idx))
        print(f"✓ Loaded {len(cond_sketches)} keyframe sketch(es)")
        
        # Load sketch masks if provided
        if args.sketch_mask:
            if len(args.sketch_mask) != len(args.sketch):
                raise ValueError("Number of sketch masks must match number of sketches")
            for mask_path, frame_idx in zip(args.sketch_mask, args.sketch_frame):
                mask_img = Image.open(mask_path).convert("L")
                sketch_masks_data.append((mask_img, frame_idx))
            print(f"✓ Loaded {len(sketch_masks_data)} sketch mask(s)")
    
    # Process conditions
    with torch.no_grad():
        masked_cond_video, preserved_cond_mask = process_conditions(
            len(cond_images), cond_images, args.num_frames, 
            target_height=target_height, target_width=target_width, device=device
        )
        
        masked_cond_sketch, preserved_sketch_mask = process_conditions(
            len(cond_sketches), cond_sketches, args.num_frames, is_sketch=True,
            target_height=target_height, target_width=target_width, device=device
        )
        
        if len(sketch_masks_data) > 0:
            sketch_local_mask = process_sketch_masks(
                sketch_masks_data, args.num_frames,
                target_height=target_height, target_width=target_width, device=device
            )
        else:
            sketch_local_mask = torch.ones((1, 1, args.num_frames, target_height, target_width), device=device)
        
        if args.fast_dev:
            print("Fast dev mode, creating dummy video")
            output_path = args.output or "output_dummy.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (target_width, target_height))
            for i in range(30):
                frame = np.full((target_height, target_width, 3), (i * 8) % 255, dtype=np.uint8)
                video_writer.write(frame)
            video_writer.release()
            print(f"✓ Dummy video saved to {output_path}")
            return output_path
        
        masked_cond_video = masked_cond_video.to(device=device, dtype=dtype)
        preserved_cond_mask = preserved_cond_mask.to(device=device, dtype=dtype)
        masked_cond_sketch = masked_cond_sketch.to(device=device, dtype=dtype)
        preserved_sketch_mask = preserved_sketch_mask.to(device=device, dtype=dtype)
        
        # Initialize latent caching if enabled
        latent_writer = None
        if args.cache_latents:
            cache_dir = args.cache_dir or f"latent_cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            latent_writer = LatentTrajectoryWriter(cache_dir, use_memmap=args.use_memmap)
            print(f"✓ Latent caching enabled: {cache_dir}")
        
        print("Running generation...")
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(device).type):
            model.pipe.device = device
            generated_video = model.pipe(
                prompt=[args.prompt],
                negative_prompt=[model.negative_prompt],
                input_image=None,
                num_inference_steps=args.num_inference_steps,
                num_frames=args.num_frames,
                seed=args.seed,
                tiled=True,
                input_condition_video=masked_cond_video,
                input_condition_preserved_mask=preserved_cond_mask,
                input_condition_video_sketch=masked_cond_sketch,
                input_condition_preserved_mask_sketch=preserved_sketch_mask,
                sketch_local_mask=sketch_local_mask,
                cfg_scale=args.cfg_scale,
                sequence_cond_residual_scale=args.sequence_cond_residual_scale,
                height=target_height,
                width=target_width,
                latent_writer=latent_writer,
            )
        
        # Convert to video
        video_frames = model.pipe.tensor2video(generated_video[0].cpu())
        
        output_path = args.output or f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        width, height = video_frames[0].size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
        
        for frame in video_frames:
            frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        
        video_writer.release()
        print(f"✓ Video saved to {output_path}")
        
    return output_path

def main():
    parser = argparse.ArgumentParser(
        description="ToonComposer CLI - Generate cartoon videos from keyframe images and sketches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with one image and sketch
  %(prog)s --prompt "A man underwater" --image img.png --image_frame 0 \\
           --sketch sketch.png --sketch_frame 30 --num_frames 61
  
  # Multiple sketches with masks at 608p
  %(prog)s --prompt "Growing flower" --image img.png --image_frame 0 \\
           --sketch s1.png --sketch_frame 30 --sketch_mask m1.png \\
           --sketch s2.png --sketch_frame 60 --sketch_mask m2.png \\
           --num_frames 61 --resolution 608p --output flower.mp4
  
  # Load sample
  %(prog)s --load_sample 1 --output sample1.mp4
        """
    )
    
    # Sample loading
    parser.add_argument("--load_sample", type=int, choices=[1, 2, 3],
                       help="Load a predefined sample (1, 2, or 3)")
    
    # Video settings
    parser.add_argument("--num_frames", type=int, default=61,
                       help="Number of frames to generate (default: 61)")
    parser.add_argument("--resolution", type=str, choices=["480p", "608p"], default="480p",
                       help="Output resolution (default: 480p)")
    parser.add_argument("--prompt", type=str,
                       help="Text prompt describing the video")
    
    # Condition inputs
    parser.add_argument("--image", type=str, action="append",
                       help="Path to keyframe image (can be specified multiple times)")
    parser.add_argument("--image_frame", type=int, action="append",
                       help="Frame index for corresponding image (can be specified multiple times)")
    parser.add_argument("--sketch", type=str, action="append",
                       help="Path to keyframe sketch (can be specified multiple times)")
    parser.add_argument("--sketch_frame", type=int, action="append",
                       help="Frame index for corresponding sketch (can be specified multiple times)")
    parser.add_argument("--sketch_mask", type=str, action="append",
                       help="Path to sketch mask image (optional, can be specified multiple times)")
    
    # Generation parameters
    parser.add_argument("--cfg_scale", type=float, default=7.5,
                       help="Classifier-free guidance scale (default: 7.5)")
    parser.add_argument("--sequence_cond_residual_scale", type=float, default=1.0,
                       help="Position-aware residual scale (default: 1.0)")
    parser.add_argument("--num_inference_steps", type=int, default=15,
                       help="Number of inference steps (default: 15)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    
    # Output
    parser.add_argument("--output", "-o", type=str,
                       help="Output video path (default: auto-generated)")
    
    # Model settings
    parser.add_argument("--device", type=str, default=os.environ.get("DEVICE", "cuda"),
                       help="Device to run on (default: cuda)")
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float32"], default="bfloat16",
                       help="Data type for inference (default: bfloat16)")
    parser.add_argument("--wan_model_dir", type=str, default=os.environ.get("WAN21_I2V_DIR"),
                       help="Local directory containing Wan2.1 model files")
    parser.add_argument("--tooncomposer_dir", type=str, default=os.environ.get("TOONCOMPOSER_DIR"),
                       help="Local directory containing ToonComposer weights")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"),
                       help="Hugging Face token for gated models")
    parser.add_argument("--fast_dev", action="store_true",
                       help="Fast dev mode (no model loading)")
    
    # Latent caching options
    parser.add_argument("--cache_latents", action="store_true",
                       help="Enable latent trajectory caching for iterative correction")
    parser.add_argument("--cache_dir", type=str,
                       help="Directory to save cached latents (default: auto-generated)")
    parser.add_argument("--use_memmap", action="store_true",
                       help="Use memory-mapped arrays for large latents (saves memory)")
    
    # Stage 2: Edit from cache options
    parser.add_argument("--edit_from_cache", type=str,
                       help="Cache directory to resume from for editing")
    parser.add_argument("--edit_mask", type=str,
                       help="Path to edit mask video/image (white=edit, black=preserve)")
    parser.add_argument("--edit_start_timestep", type=int,
                       help="Timestep to start editing from (default: full denoising)")
    parser.add_argument("--edit_feather_size", type=int, default=5,
                       help="Mask feathering kernel size (default: 5, use 0 to disable)")
    parser.add_argument("--edit_feather_sigma", type=float, default=2.0,
                       help="Mask feathering sigma (default: 2.0)")
    
    args = parser.parse_args()
    
    # Setup
    dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    
    # Resolve directories and initialize
    print("Initializing ToonComposer...")
    toon_dir = resolve_tooncomposer_repo_dir(preferred_dir=args.tooncomposer_dir, hf_token=args.hf_token)
    checkpoints_by_resolution = build_checkpoints_by_resolution(toon_dir)
    
    model, device, dtype = initialize_model(
        resolution=args.resolution,
        fast_dev=args.fast_dev,
        device=args.device,
        dtype=dtype,
        wan_model_dir=args.wan_model_dir,
        tooncomposer_dir=args.tooncomposer_dir,
        hf_token=args.hf_token,
        checkpoints_by_resolution=checkpoints_by_resolution
    )
    
    # Generate video
    try:
        output_path = generate_video(args, model, device, dtype, checkpoints_by_resolution)
        print(f"\n✓ Success! Video generated: {output_path}")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()
