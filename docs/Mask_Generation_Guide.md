# Mask Generation Guide

Generate video masks for localized editing in ToonComposer. White regions (255) = edit, Black regions (0) = preserve.

## Quick Start

```bash
python util/generate_mask.py \
  --width 1024 --height 576 \
  --fps 24 --frames 100 \
  --white "0-30" \
  --output mask.mp4
```

## Command-Line Tool

### Basic Parameters

```bash
python util/generate_mask.py \
  -W <width> -H <height> \
  -r <fps> -n <frames> \
  -o <output.mp4>
```

**Required:**
- `-W, --width`: Frame width
- `-H, --height`: Frame height  
- `-r, --fps`: Frame rate
- `-n, --frames`: Total frames

**Optional:**
- `--white`: White frame ranges (e.g., `"0-10, 50-60"`)
- `--black`: Black frame ranges
- `--default`: Default color (`black` or `white`)
- `--regions`: JSON regions (see below)
- `--overwrite`: Overwrite existing file

### Examples

**Full-frame mask:**
```bash
python util/generate_mask.py -W 1024 -H 576 -r 24 -n 100 \
  --white "0-20, 50-60" -o mask.mp4
```

**Rectangle region:**
```bash
python util/generate_mask.py -W 1024 -H 576 -r 24 -n 100 \
  --regions '[{"type":"rect","coords":[100,100,400,300],"frames":"0-50"}]' \
  -o rect_mask.mp4
```

**Polygon region:**
```bash
python util/generate_mask.py -W 1024 -H 576 -r 24 -n 100 \
  --regions '[{"type":"polygon","coords":[[100,100],[500,100],[300,400]],"frames":"10-40"}]' \
  -o poly_mask.mp4
```

## Region JSON Format

### Quick Reference

```json
[
  {
    "type": "rect",
    "coords": [x, y, width, height],
    "frames": "0-30, 50-60",
    "color": "white"
  },
  {
    "type": "polygon",
    "coords": [[x1,y1], [x2,y2], [x3,y3]],
    "frames": "15-45",
    "color": "white"
  }
]
```

- **type**: `"rect"` or `"polygon"`
- **coords**: `[x, y, w, h]` for rect, `[[x,y],...]` for polygon
- **frames**: Frame ranges (0-indexed, inclusive)
- **color**: `"white"` or `"black"`

### Step-by-Step Guide

#### Step 1: Determine Your Video Dimensions
Know your video size. Example: 1024×576 pixels
- x coordinates: 0 to 1023
- y coordinates: 0 to 575

#### Step 2: Choose Region Type

**Rectangle** - Simple box region:
```json
{
  "type": "rect",
  "coords": [x, y, width, height]
}
```
- `x, y`: Top-left corner position
- `width, height`: Size of the box

**Example:** 200×150 box starting at position (100, 100):
```json
{
  "type": "rect",
  "coords": [100, 100, 200, 150]
}
```

**Polygon** - Custom shape with corners:
```json
{
  "type": "polygon",
  "coords": [[x1, y1], [x2, y2], [x3, y3], ...]
}
```
- List each corner point in order
- Will automatically close (connect last to first)

**Example:** Triangle:
```json
{
  "type": "polygon",
  "coords": [[100, 100], [500, 100], [300, 400]]
}
```

#### Step 3: Specify Frame Ranges

**Single range:**
```json
"frames": "0-30"
```
Applies to frames 0, 1, 2, ..., 30 (inclusive)

**Multiple ranges:**
```json
"frames": "0-30, 50-60, 80-90"
```

**Individual frames:**
```json
"frames": "5, 10, 15, 20"
```

**Mixed:**
```json
"frames": "0-10, 15, 20-25, 30"
```

#### Step 4: Set Color (Optional)

```json
"color": "white"  // Edit this region
"color": "black"  // Keep this region unchanged
```
Default is `"white"` if not specified.

#### Step 5: Combine Multiple Regions

Save as `regions.json`:
```json
[
  {
    "type": "rect",
    "coords": [100, 100, 300, 200],
    "frames": "0-30",
    "color": "white"
  },
  {
    "type": "polygon",
    "coords": [[500, 200], [700, 200], [600, 400]],
    "frames": "20-50",
    "color": "white"
  }
]
```

Then use:
```bash
python util/generate_mask.py -W 1024 -H 576 -r 24 -n 100 \
  --regions regions.json -o mask.mp4
```

### Common Patterns

**Full-frame for specific frames:**
```json
[
  {
    "type": "rect",
    "coords": [0, 0, 1024, 576],
    "frames": "0-30"
  }
]
```

**Top half of video:**
```json
[
  {
    "type": "rect",
    "coords": [0, 0, 1024, 288],
    "frames": "0-100"
  }
]
```

**Center square:**
```json
[
  {
    "type": "rect",
    "coords": [312, 88, 400, 400],
    "frames": "0-100"
  }
]
```

**Moving region (multiple time ranges):**
```json
[
  {
    "type": "rect",
    "coords": [100, 100, 200, 200],
    "frames": "0-30"
  },
  {
    "type": "rect",
    "coords": [400, 100, 200, 200],
    "frames": "31-60"
  }
]
```

## Gradio API (Interactive)

```python
from util.mask_generator_api import generate_mask_from_scribbles

generate_mask_from_scribbles(
    width=1024, height=576, fps=24, total_frames=100,
    scribbles=[{
        "mask": mask_array,  # np.array (H,W) with 0/255
        "start_frame": 0,
        "end_frame": 30
    }],
    output_path="mask.mp4"
)
```

## Tips

- Match mask resolution to video resolution
- White = areas to edit, Black = areas to preserve
- Use `--overwrite` to replace existing masks
- Try `--codec XVID` if default codec fails

## Related

- `util/generate_mask.py` - CLI tool
- `util/mask_generator_api.py` - Gradio API
- `util/mask_utils.py` - Mask processing
- [Iterative Correction System](Iterative_Correction_System.md)
