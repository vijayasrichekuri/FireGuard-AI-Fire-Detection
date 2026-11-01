import os
import json
import numpy as np
from PIL import Image

def create_sample_dataset():
    os.makedirs("datasets/train", exist_ok=True)
    os.makedirs("datasets/annotations", exist_ok=True)
    
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "fire"}]
    }
    
    for i in range(5):
        img_array = np.zeros((400, 400, 3), dtype=np.uint8)
        img_array[100:300, 100:300] = [255, 0, 0]
        
        img = Image.fromarray(img_array)
        filename = f"sample_fire_{i+1}.jpg"
        img.save(f"datasets/train/{filename}")
        
        image_id = i + 1
        coco_format["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": 400,
            "height": 400
        })
        
        coco_format["annotations"].append({
            "id": i + 1,
            "image_id": image_id,
            "category_id": 1,
            "segmentation": [[100, 100, 300, 100, 300, 300, 100, 300]],
            "area": 200 * 200,
            "bbox": [100, 100, 200, 200],
            "iscrowd": 0
        })
    
    with open('datasets/annotations/train.json', 'w') as f:
        json.dump(coco_format, f, indent=2)
    
    print("Created sample dataset with 5 fire images")

if __name__ == "__main__":
    create_sample_dataset()
