"""API for generating mask videos from Gradio scribbles"""
import numpy as np
import cv2
from typing import List, Tuple

def generate_mask_from_scribbles(
    width: int,
    height: int,
    fps: float,
    total_frames: int,
    scribbles: List[dict],  # [{"mask": np.array, "start_frame": int, "end_frame": int}, ...]
    output_path: str,
    default_color: str = "black"
) -> str:
    """
    Generate mask video from Gradio scribbles.
    
    Args:
        width, height: Video dimensions
        fps: Frame rate
        total_frames: Total number of frames
        scribbles: List of scribble dicts, each containing:
            - mask: numpy array (H, W) with painted regions (0 or 255)
            - start_frame: Starting frame index
            - end_frame: Ending frame index (inclusive)
        output_path: Output video file path
        default_color: "black" or "white"
    
    Returns:
        Path to generated video
    """
    default_value = 255 if default_color == "white" else 0
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_idx in range(total_frames):
        frame = np.full((height, width, 3), default_value, dtype=np.uint8)
        
        # Apply all scribbles that affect this frame
        for scribble in scribbles:
            if scribble["start_frame"] <= frame_idx <= scribble["end_frame"]:
                mask = scribble["mask"]
                # Resize mask if needed
                if mask.shape[:2] != (height, width):
                    mask = cv2.resize(mask, (width, height))
                
                # Apply white where mask is non-zero
                frame[mask > 0] = 255
        
        writer.write(frame)
    
    writer.release()
    return output_path
