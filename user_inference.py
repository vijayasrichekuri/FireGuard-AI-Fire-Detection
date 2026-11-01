import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def get_transform():
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

class FireDetector:
    def __init__(self, model_path):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = get_model(2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded successfully!")
        
    def detect(self, image_path, confidence=0.5):
        try:
            image = Image.open(image_path).convert("RGB")
            original_image = np.array(image)
            transform = get_transform()
            image_tensor = transform(image=original_image)['image']
            
            with torch.no_grad():
                prediction = self.model([image_tensor.to(self.device)])
            
            pred = prediction[0]
            scores = pred['scores'].cpu().numpy()
            masks = pred['masks'].cpu().numpy()
            boxes = pred['boxes'].cpu().numpy()
            
            high_conf = scores > confidence
            boxes = boxes[high_conf]
            masks = masks[high_conf]
            scores = scores[high_conf]
            
            return {
                'boxes': boxes,
                'masks': masks, 
                'scores': scores,
                'original_image': original_image
            }
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            return None
    
    def visualize(self, results, save_path=None):
        if results is None:
            return
            
        image = results['original_image']
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(image)
        
        if len(results['boxes']) > 0:
            for i, (box, mask, score) in enumerate(zip(results['boxes'], results['masks'], results['scores'])):
                x1, y1, x2, y2 = box.astype(int)
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
                axes[1].add_patch(rect)
                
                mask = mask[0]
                axes[1].imshow(mask, alpha=0.3, cmap='jet')
                
                axes[1].text(x1, y1-10, f'Fire: {score:.2f}', color='red', fontsize=12, 
                           bbox=dict(boxstyle="round", fc="white", alpha=0.7))
            
            print(f"🔥 Detected {len(results['boxes'])} fire regions")
        else:
            axes[1].text(0.5, 0.5, 'No Fire Detected', horizontalalignment='center', 
                        verticalalignment='center', transform=axes[1].transAxes, 
                        fontsize=16, color='green', bbox=dict(boxstyle="round", fc="lightgreen"))
            print("✅ No fire detected - Image is safe")
        
        axes[1].set_title('Fire Detection Results')
        axes[1].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Results saved to: {save_path}")
        plt.show()

def get_user_image():
    print("\n" + "="*50)
    print("🔥 FIRE DETECTION SYSTEM")
    print("="*50)
    
    while True:
        print("\nChoose an option:")
        print("1. Enter image file path")
        print("2. Use sample images from datasets/train/")
        print("3. Use webcam to capture image")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            image_path = input("Enter the full path to your image: ").strip()
            if os.path.exists(image_path):
                return image_path
            else:
                print("❌ File not found. Please check the path.")
                
        elif choice == '2':
            if os.path.exists("datasets/train"):
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    image_files.extend([f for f in os.listdir('datasets/train') if f.lower().endswith(ext[1:])])
                
                if image_files:
                    print("\nAvailable sample images:")
                    for i, img in enumerate(image_files):
                        print(f"{i+1}. {img}")
                    
                    try:
                        img_choice = int(input(f"\nSelect image (1-{len(image_files)}): ")) - 1
                        if 0 <= img_choice < len(image_files):
                            return os.path.join("datasets/train", image_files[img_choice])
                        else:
                            print("❌ Invalid selection.")
                    except ValueError:
                        print("❌ Please enter a valid number.")
                else:
                    print("❌ No sample images found in datasets/train/")
            else:
                print("❌ datasets/train/ folder not found.")
                
        elif choice == '3':
            return capture_from_webcam()
            
        elif choice == '4':
            return None
            
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

def capture_from_webcam():
    print("\n📷 Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot access webcam")
        return None
    
    print("Press SPACE to capture image, ESC to cancel")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture image")
            break
            
        cv2.imshow('Webcam - Press SPACE to capture', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            os.makedirs("captured_images", exist_ok=True)
            image_path = "captured_images/webcam_capture.jpg"
            cv2.imwrite(image_path, frame)
            print(f"✅ Image captured and saved as: {image_path}")
            break
        elif key == 27:  # ESC
            print("❌ Capture cancelled")
            image_path = None
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return image_path

def main():
    # Check if model exists
    if not os.path.exists("models/fire_detection_model.pth"):
        print("❌ Model not found! Please run train.py first.")
        return
    
    # Load detector
    detector = FireDetector("models/fire_detection_model.pth")
    
    while True:
        image_path = get_user_image()
        
        if image_path is None:
            print("👋 Thank you for using Fire Detection System!")
            break
            
        print(f"\n🖼️  Processing: {image_path}")
        
        # Get confidence threshold from user
        try:
            confidence = float(input("Enter confidence threshold (0.1-0.9, default 0.5): ") or "0.5")
            confidence = max(0.1, min(0.9, confidence))  # Clamp between 0.1 and 0.9
        except ValueError:
            confidence = 0.5
            print("Using default confidence: 0.5")
        
        # Detect fire
        results = detector.detect(image_path, confidence=confidence)
        
        if results is not None:
            # Ask if user wants to save results
            save_choice = input("Save results? (y/n, default n): ").strip().lower()
            save_path = None
            if save_choice in ['y', 'yes']:
                os.makedirs("results", exist_ok=True)
                filename = os.path.basename(image_path)
                save_path = f"results/detection_{filename}"
            
            # Show results
            detector.visualize(results, save_path)
        
        # Ask if user wants to continue
        continue_choice = input("\nProcess another image? (y/n, default y): ").strip().lower()
        if continue_choice in ['n', 'no']:
            print("👋 Thank you for using Fire Detection System!")
            break

if __name__ == "__main__":
    main()
