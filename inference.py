import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import cv2
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
        
    def detect(self, image_path, confidence=0.5):
    image = Image.open(image_path).convert("RGB")
    original_image = np.array(image)
    transform = get_transform()
    image_tensor = transform(image=original_image)['image']
    
    # Remove batch dimension if present
    if image_tensor.dim() == 4:  # [1, 3, 512, 512]
        image_tensor = image_tensor.squeeze(0)  # becomes [3, 512, 512]
    
    with torch.no_grad():
        prediction = self.model([image_tensor.to(self.device)])  # Now correct!
    
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
    
    def visualize(self, results, save_path=None):
        image = results['original_image']
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(image)
        for i, (box, mask, score) in enumerate(zip(results['boxes'], results['masks'], results['scores'])):
            x1, y1, x2, y2 = box.astype(int)
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
            axes[1].add_patch(rect)
            
            mask = mask[0]
            axes[1].imshow(mask, alpha=0.3, cmap='jet')
            
            axes[1].text(x1, y1-10, f'Fire: {score:.2f}', color='red', fontsize=12, bbox=dict(boxstyle="round", fc="white", alpha=0.7))
        
        axes[1].set_title('Fire Detection Results')
        axes[1].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results saved to: {save_path}")
        plt.show()

if __name__ == "__main__":
    detector = FireDetector('models/fire_detection_model.pth')
    results = detector.detect('datasets/train/sample_fire_1.jpg')
    detector.visualize(results, 'results/detection_output.png')
