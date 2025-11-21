from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image
from missing_region import *

pipeline = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float32).to("cuda")

def create_mask(width, height, size=0.3, position=(0, 0)):
    """Create a mask image (black background with white region to inpaint)"""
    missing_width = int(width * size)
    missing_height = int(height * size)
    white_region = Image.new("RGB", (missing_width, missing_height), (255, 255, 255))
    mask = Image.new("RGB", (width, height), (0, 0, 0))  # Black background (RGB)
    mask.paste(white_region, position)
    return mask

def inpaint_image(image, mask, prompt="", position=(0, 0)):
    # Image should already be 512x512, but ensure it is
    if image.size != (512, 512):
        image = image.resize((512, 512))
    # Ensure image is RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Create the damaged image with white region
    damaged_image, _ = create_missing_region(image.copy(), position=position)
    # Ensure damaged image is RGB
    if damaged_image.mode != "RGB":
        damaged_image = damaged_image.convert("RGB")
    # Ensure mask matches the image size exactly and is RGB
    if mask.size != image.size:
        mask = mask.resize(image.size)
    if mask.mode != "RGB":
        mask = mask.convert("RGB")
    image = pipeline(prompt, image=damaged_image, mask_image=mask, num_inference_steps=100, guidance_scale=7.5).images[0]
    return image

if __name__ == "__main__":
    image = Image.open("pikachu.png")
    position = (100, 100)
    # Resize to 512x512 first
    # print(image.size)
    image_resized = image.resize((512, 512))
    mask = create_mask(512, 512, position=position)
    damaged_image, _ = create_missing_region(image_resized.copy(), position=position)
    damaged_image.save("results/test_images/pikachu_damaged.png")
    mask.save("results/test_images/pikachu_mask.png")
    image = inpaint_image(image_resized, mask, prompt="a pikachu", position=position)
    image.save("results/test_images/pikachu_inpainted.png")