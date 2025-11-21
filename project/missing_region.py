import numpy as np
from PIL import Image

def create_missing_region(image, size = 0.3, position=(0, 0)):
    width, height = image.size
    missing_width = int(width * size)
    missing_height = int(height * size)
    missing_region = Image.new("RGB", (missing_width, missing_height), (255, 255, 255))
    position = (100, 100)
    image.paste(missing_region, position)
    return image, position
    # return missing_region

if __name__ == "__main__":
    image = Image.open("pikachu.png")
    image, _ = create_missing_region(image)
    image.save("results/test_images/pikachu_missing.png")