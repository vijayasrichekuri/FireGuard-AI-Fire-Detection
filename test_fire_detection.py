import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def test_fire_detection(image_path):
    """Test fire detection on a specific image"""
    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image)
    
    # Test different color spaces
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    
    # Fire color ranges
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    lower_orange = np.array([11, 50, 50])
    upper_orange = np.array([25, 255, 255])
    
    # Create masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask3 = cv2.inRange(hsv, lower_orange, upper_orange)
    
    combined_mask = mask1 + mask2 + mask3
    
    # Calculate statistics
    total_pixels = img_array.shape[0] * img_array.shape[1]
    fire_pixels = np.count_nonzero(combined_mask)
    fire_ratio = fire_pixels / total_pixels
    
    print(f"Image: {image_path}")
    print(f"Total pixels: {total_pixels}")
    print(f"Fire pixels: {fire_pixels}")
    print(f"Fire ratio: {fire_ratio:.4f} ({fire_ratio*100:.2f}%)")
    print(f"Should detect: {'YES' if fire_ratio > 0.005 else 'NO'}")
    print("-" * 50)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_array)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(combined_mask, cmap='hot')
    axes[1].set_title("Fire Mask")
    axes[1].axis('off')
    
    result_img = img_array.copy()
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 50:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    axes[2].imshow(result_img)
    axes[2].set_title("Detected Fire Regions")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test on your fire image
    test_fire_detection("path_to_your_fire_image.jpg")
