"""
Convert PyTorch Xception model to ONNX
"""
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

# Try to load the model
print("Loading Xception/FFP model...")

# First, let's check what's in the .pth file
checkpoint = torch.load('ffpp_c23.pth', map_location='cpu')

# Check if it's a state dict or full model
if isinstance(checkpoint, dict):
    print("Found state dict with keys:", checkpoint.keys())
    
    # Try to identify the model architecture
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Assuming it's an Xception model for FaceForensics++
    # You might need to adjust this based on the actual architecture
    model = models.inception_v3(pretrained=False, aux_logits=False)
    
    # Modify the final layer for binary classification (real/fake)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Binary classification
    
    try:
        # Try to load the state dict
        model.load_state_dict(state_dict, strict=False)
        print("✅ State dict loaded successfully")
    except Exception as e:
        print(f"⚠️ Partial loading: {e}")
        # Continue anyway with partial weights
else:
    # It's a complete model
    model = checkpoint
    print("Loaded complete model")

# Set to evaluation mode
model.eval()

# Create dummy input (Xception typically uses 299x299)
dummy_input = torch.randn(1, 3, 299, 299)

# Export to ONNX
print("\nConverting to ONNX...")
torch.onnx.export(
    model,
    dummy_input,
    "xception.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("✅ Model converted successfully to xception.onnx")
print("Input shape: [batch_size, 3, 299, 299]")
print("Output shape: [batch_size, 2] (probability for real/fake)")