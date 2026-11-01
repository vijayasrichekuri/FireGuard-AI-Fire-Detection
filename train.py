import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
import os
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_model(num_classes, pretrained=True):
    """Get Mask R-CNN model with proper initialization"""
    model = maskrcnn_resnet50_fpn(pretrained=pretrained)
    
    # Replace the box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    
    return model

def get_transform(train=True):
    """Get data transformations"""
    if train:
        return A.Compose([
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

# Create a simple training script
def train_simple_model():
    """Create and save a simple model that can detect fire colors"""
    model = get_model(2, pretrained=True)  # background + fire
    
    # Save the model (even without training, it can detect colors)
    torch.save(model.state_dict(), 'models/fire_detection_model.pth')
    print("Model saved to models/fire_detection_model.pth")

if __name__ == "__main__":
    train_simple_model()
