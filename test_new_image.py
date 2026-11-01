import numpy as np
from PIL import Image
import os

def create_test_image():
    os.makedirs("test_images", exist_ok=True)
    
    # Create a test image with fire-like region
    img_array = np.zeros((400, 400, 3), dtype=np.uint8)
    # Add red rectangle (fire)
    img_array[150:250, 150:250] = [255, 100, 0]  # Orange-red color
    
    img = Image.fromarray(img_array)
    img.save("test_images/test_fire.jpg")
    print("Created test image: test_images/test_fire.jpg")

if __name__ == "__main__":
    create_test_image()
