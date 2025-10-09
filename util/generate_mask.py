
import argparse
import os
import re
import sys
from typing import Set
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate a video with specified resolution, fps, frame count, and frame ranges "
                    "that are pure white or pure black. Ranges are inclusive and 0-based."
    )
    parser.add_argument("--width", "-W", type=int, required=True, help="Frame width in pixels.")
    parser.add_argument("--height", "-H", type=int, required=True, help="Frame height in pixels.")
    parser.add_argument("--fps", "-r", type=float, required=True, help="Frames per second.")
    parser.add_argument("--frames", "-n", type=int, required=True, help="Total number of frames.")
    parser.add_argument("--white", type=str, default="", help='White frame ranges, e.g. "0-10, 50-60".')
    parser.add_argument("--black", type=str, default="", help='Black frame ranges, e.g. "11-20, 70-80".')
    parser.add_argument("--default", choices=["black", "white"], default="black",
                        help="Color for unspecified frames.")
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

    # Parse ranges
    try:
        white_set = parse_ranges(args.white, args.frames)
        black_set = parse_ranges(args.black, args.frames)
    except ValueError as e:
        print(f"Invalid range specification: {e}")
        sys.exit(4)

    # Compute precedence: white overrides black
    conflict = white_set & black_set
    if conflict:
        # Remove conflicts from black so that white wins
        black_set -= conflict

    # Summary
    print(f"Generating video: {out_path}")
    print(f"Resolution: {args.width}x{args.height} @ {args.fps} fps, frames: {args.frames}")
    print(f"White frames: {len(white_set)}; Black frames: {len(black_set)}; Default: {args.default}")
    if conflict:
        print(f"Note: {len(conflict)} overlapping frames resolved in favor of white.")

    # Prepare frames
    h, w = args.height, args.width
    white_frame = np.full((h, w, 3), 255, dtype=np.uint8)
    black_frame = np.zeros((h, w, 3), dtype=np.uint8)
    default_frame = white_frame if args.default == "white" else black_frame

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(out_path, fourcc, args.fps, (w, h))
    if not writer.isOpened():
        print("Failed to open VideoWriter. Try a different --codec or output extension.")
        sys.exit(5)

    try:
        for i in range(args.frames):
            if i in white_set:
                frame = white_frame
            elif i in black_set:
                frame = black_frame
            else:
                frame = default_frame
            writer.write(frame)
    finally:
        writer.release()

    print("Done.")


if __name__ == "__main__":
    main()
