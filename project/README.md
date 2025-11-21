# Image Inpainting Project

This project implements image inpainting using Stable Diffusion.

## Files

- `inpainting.py` - Main inpainting script using Stable Diffusion Inpaint Pipeline
- `missing_region.py` - Function to create missing/damaged regions in images

## Requirements

- Python 3.10+
- PyTorch
- diffusers
- transformers
- accelerate
- Pillow (PIL)
- CUDA-capable GPU

## Installation

```bash
conda activate genai_proj
pip install diffusers transformers accelerate pillow torch
```

## Usage

### Basic Inpainting

```python
from inpainting import inpaint_image, create_mask
from PIL import Image

# Load image
image = Image.open("pikachu.png")
image = image.resize((512, 512))

# Create mask
mask = create_mask(512, 512, size=0.3, position=(0, 0))

# Inpaint
result = inpaint_image(image, mask, prompt="a pikachu")
result.save("result.png")
```

### Creating Missing Regions

```python
from missing_region import create_missing_region
from PIL import Image

image = Image.open("pikachu.png")
damaged_image, position = create_missing_region(image, size=0.3, position=(100, 100))
damaged_image.save("damaged.png")
```

## Notes

- Images are automatically resized to 512x512 for Stable Diffusion
- The pipeline uses float32 precision for compatibility
- Masks should be RGB format with white regions indicating areas to inpaint

