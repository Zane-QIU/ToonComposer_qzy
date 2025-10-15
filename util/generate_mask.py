import argparse
import json
import os
import re
import sys
from typing import Set, List, Dict, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    print("OpenCV (cv2) is required. Install with: pip install opencv-python numpy")
    sys.exit(1)


def parse_ranges(spec: str, total_frames: int) -> Set[int]:
    """
    Parse a range string like "0-10, 15, 20-22" into a set of frame indices.
    - Ranges are inclusive.
    - Indices are 0-based.
    - Out-of-bound indices are ignored.
    """
    result: Set[int] = set()
    if not spec:
        return result
    parts = re.split(r"[,\s]+", spec.strip())
    for part in parts:
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                start = int(a)
                end = int(b)
            except ValueError:
                raise ValueError(f"Invalid range token: '{part}'")
            step = 1 if end >= start else -1
            for i in range(start, end + step, step):
                if 0 <= i < total_frames:
                    result.add(i)
        else:
            try:
                i = int(part)
            except ValueError:
                raise ValueError(f"Invalid index token: '{part}'")
            if 0 <= i < total_frames:
                result.add(i)
    return result


def parse_mask_regions(regions_json: str) -> List[Dict]:
    """
    Parse mask regions from JSON string or file.
    
    Expected format:
    [
        {
            "type": "rect",  # or "polygon"
            "coords": [x, y, w, h],  # for rect: [x, y, width, height]
                                      # for polygon: [[x1,y1], [x2,y2], ...]
            "frames": "0-10, 15-20",  # frame range string
            "color": "white"  # "white" or "black", default "white"
        },
        ...
    ]
    """
    if not regions_json:
        return []
    
    # Check if it's a file path
    if os.path.isfile(regions_json):
        with open(regions_json, 'r') as f:
            data = json.load(f)
    else:
        data = json.loads(regions_json)
    
    if not isinstance(data, list):
        raise ValueError("Mask regions must be a JSON array")
    
    return data


def apply_mask_region(frame: np.ndarray, region: Dict, color_value: int = 255):
    """
    Apply a mask region to a frame.
    
    Args:
        frame: The frame to modify (will be modified in-place)
        region: Region specification dict
        color_value: 255 for white, 0 for black
    """
    region_type = region.get("type", "rect").lower()
    coords = region.get("coords")
    
    if region_type == "rect":
        # coords: [x, y, width, height]
        if len(coords) != 4:
            raise ValueError(f"Rectangle coords must have 4 values: {coords}")
        x, y, w, h = coords
        frame[y:y+h, x:x+w] = color_value
        
    elif region_type == "polygon":
        # coords: [[x1,y1], [x2,y2], ...]
        pts = np.array(coords, dtype=np.int32)
        cv2.fillPoly(frame, [pts], color=(color_value, color_value, color_value))
        
    else:
        raise ValueError(f"Unknown region type: {region_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a video mask with spatial-temporal regions. "
                    "Supports both full-frame ranges and specific spatial regions across time."
    )
    parser.add_argument("--width", "-W", type=int, required=True, help="Frame width in pixels.")
    parser.add_argument("--height", "-H", type=int, required=True, help="Frame height in pixels.")
    parser.add_argument("--fps", "-r", type=float, required=True, help="Frames per second.")
    parser.add_argument("--frames", "-n", type=int, required=True, help="Total number of frames.")
    
    # Legacy full-frame options (kept for backward compatibility)
    parser.add_argument("--white", type=str, default="", help='Full-frame white ranges, e.g. "0-10, 50-60".')
    parser.add_argument("--black", type=str, default="", help='Full-frame black ranges, e.g. "11-20, 70-80".')
    
    # New spatial-temporal region options
    parser.add_argument("--regions", type=str, default="", 
                        help='JSON string or file path defining mask regions. '
                             'Format: [{"type":"rect","coords":[x,y,w,h],"frames":"0-10","color":"white"},...]')
    
    parser.add_argument("--default", choices=["black", "white"], default="black",
                        help="Color for unspecified frames/regions.")
    parser.add_argument("--codec", type=str, default="mp4v",
                        help="FourCC codec (e.g., mp4v, XVID, MJPG). Default: mp4v")
    parser.add_argument("--output", "-o", type=str, default="output.mp4", help="Output video file path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists.")
    args = parser.parse_args()

    # Basic validation
    if args.width <= 0 or args.height <= 0:
        print("Width and height must be positive integers.")
        sys.exit(2)
    if args.fps <= 0:
        print("FPS must be > 0.")
        sys.exit(2)
    if args.frames <= 0:
        print("Total frames must be > 0.")
        sys.exit(2)

    # Prepare output
    out_path = args.output
    if os.path.exists(out_path) and not args.overwrite:
        print(f"Output file exists: {out_path}. Use --overwrite to replace.")
        sys.exit(3)

    # Parse legacy full-frame ranges
    try:
        white_set = parse_ranges(args.white, args.frames)
        black_set = parse_ranges(args.black, args.frames)
    except ValueError as e:
        print(f"Invalid range specification: {e}")
        sys.exit(4)

    # Parse spatial-temporal regions
    try:
        mask_regions = parse_mask_regions(args.regions)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Invalid mask regions: {e}")
        sys.exit(4)

    # Process mask regions: organize by frame
    frame_regions: Dict[int, List[Dict]] = {}
    for region in mask_regions:
        frames_str = region.get("frames", "")
        region_frames = parse_ranges(frames_str, args.frames)
        color = region.get("color", "white").lower()
        
        for frame_idx in region_frames:
            if frame_idx not in frame_regions:
                frame_regions[frame_idx] = []
            frame_regions[frame_idx].append({
                "type": region.get("type", "rect"),
                "coords": region.get("coords"),
                "color": color
            })

    # Compute precedence for legacy options: white overrides black
    conflict = white_set & black_set
    if conflict:
        black_set -= conflict

    # Summary
    print(f"Generating video: {out_path}")
    print(f"Resolution: {args.width}x{args.height} @ {args.fps} fps, frames: {args.frames}")
    print(f"Legacy - White frames: {len(white_set)}; Black frames: {len(black_set)}")
    print(f"Spatial regions: {len(mask_regions)} regions defined")
    print(f"Default: {args.default}")
    if conflict:
        print(f"Note: {len(conflict)} overlapping frames resolved in favor of white.")

    # Prepare frames
    h, w = args.height, args.width
    default_value = 255 if args.default == "white" else 0

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(out_path, fourcc, args.fps, (w, h))
    if not writer.isOpened():
        print("Failed to open VideoWriter. Try a different --codec or output extension.")
        sys.exit(5)

    try:
        for i in range(args.frames):
            # Start with default color
            frame = np.full((h, w, 3), default_value, dtype=np.uint8)
            
            # Apply legacy full-frame settings
            if i in white_set:
                frame[:] = 255
            elif i in black_set:
                frame[:] = 0
            
            # Apply spatial regions (these override legacy settings)
            if i in frame_regions:
                for region in frame_regions[i]:
                    color_val = 255 if region["color"] == "white" else 0
                    try:
                        apply_mask_region(frame, region, color_val)
                    except Exception as e:
                        print(f"Warning: Failed to apply region at frame {i}: {e}")
            
            writer.write(frame)
            
            # Progress indicator
            if (i + 1) % 30 == 0 or i == args.frames - 1:
                print(f"Progress: {i + 1}/{args.frames} frames", end='\r')
        
        print()  # New line after progress
    finally:
        writer.release()

    print("Done.")


if __name__ == "__main__":
    main()
