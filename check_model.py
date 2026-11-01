# check_model.py
import torch
checkpoint = torch.load('models/fire_detection_model.pth', map_location='cpu')
keys = list(checkpoint.keys())

print(f"Total keys: {len(keys)}")
print("First 10 keys:", keys[:10])
print("Last 5 keys:", keys[-5:])

# Check for common model weight keys
important_keys = [k for k in keys if 'state_dict' in k.lower() or 'model' in k.lower() or 'weight' in k.lower() or 'backbone' in k.lower()]
print("Important keys:", important_keys)

# Check the type of the first value
if keys:
    first_key = keys[0]
    print(f"Type of '{first_key}': {type(checkpoint[first_key])}")
    print(f"Shape of '{first_key}': {checkpoint[first_key].shape if hasattr(checkpoint[first_key], 'shape') else 'No shape'}")
